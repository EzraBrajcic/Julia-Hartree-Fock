using LinearAlgebra
using CUDA

include("STOFuncs.jl")
include("DataTypes.jl")

function RealYNormalization(l::Int, m::Int)

    # Compute real spherical-harmonic normalization on CPU
    # N_l^m = sqrt((2l+1)/(4π) * (l-m)!/(l+m)!)
    num = factorial(big(l - abs(m)))
    den = factorial(big(l + abs(m)))
    val = sqrt( (2l + 1) / (4 * π) * float(num / den) )
    return Float32(val)
end

function PackPrimitives(Basis::BasisSet)

    # PackPrimitives: flatten primitive parameters into dense arrays for device
    # returns arrays NBasis x MaxPrims column-major and MOcoeff vector
    # Also packs precomputed YNorm factor for each primitive (real spherical harmonic normalization)
    NBasis = Basis.BasisSize
    Counts = Int[]
    for b in 1:NBasis
        push!(Counts, length(Basis.BasisFunctions[b].STOs))
    end
    MaxPrims = maximum(Counts)
    PrimN     = zeros(Int32, NBasis, MaxPrims)
    PrimL     = zeros(Int32, NBasis, MaxPrims)
    PrimM     = zeros(Int32, NBasis, MaxPrims)
    Primζ  = zeros(Float32, NBasis, MaxPrims)
    PrimNorm  = zeros(Float32, NBasis, MaxPrims)
    PrimCoeff = zeros(Float32, NBasis, MaxPrims)
    PrimYNorm = zeros(Float32, NBasis, MaxPrims)  # real spherical harmonic normalization N_l^m
    PrimCount = Int32.(Counts)

    for b in 1:NBasis
        stolist = Basis.BasisFunctions[b].STOs
        for p in 1:lastindex(stolist)
            sto = stolist[p]
            PrimN[b,p]     = Int32(sto.n)
            PrimL[b,p]     = Int32(sto.l)
            PrimM[b,p]     = Int32(sto.m)
            Primζ[b,p]  = Float32(sto.ζ)
            PrimNorm[b,p]  = Float32(sto.normconst)
            PrimCoeff[b,p] = Float32(sto.coeff)
            PrimYNorm[b,p] = RealYNormalization(sto.l, sto.m)
        end
    end
    return MaxPrims, PrimN, PrimL, PrimM, Primζ, PrimNorm, PrimCoeff, PrimYNorm, PrimCount
end

function KernelComputeAOWithAng!(
        AODev, XDev, YDev, ZDev,
        PrimNDev, PrimLDev, PrimMDev,
        PrimζDev, PrimNormDev, PrimCoeffDev, PrimYNormDev,
        PrimCountDev,
        NBasis::Int32, MaxPrims::Int32, NPoints::Int32
        )

    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    total = Int(NBasis) * Int(NPoints)
    if idx <= total
        j = (idx - 1) % NPoints + 1   # point index
        b = (idx - 1) ÷ NPoints + 1   # basis index

        x = XDev[j]; y = YDev[j]; z = ZDev[j]
        r = sqrt(x*x + y*y + z*z)

        # compute angular coords safely
        cosθ = 0.0f0
        sinθ = 0.0f0
        φ = 0.0f0
        if r > 0.0f0
            cosθ = z / r
            # clamp cosθ to [-1,1] to avoid NaNs from rounding
            if cosθ > 1.0f0
                cosθ = 1.0f0
            elseif cosθ < -1.0f0
                cosθ = -1.0f0
            end
            sinθ = sqrt(max(0.0f0, 1.0f0 - cosθ * cosθ))
            φ = atan(y, x)  # atan(y,x) is supported on GPU
        else
            # at origin define cosθ = 1, sinθ = 0, φ = 0
            cosθ = 1.0f0
            sinθ = 0.0f0
            φ = 0.0f0
        end

        val_total = 0.0f0
        np = PrimCountDev[b]

        @inbounds for p in 1:np
            n_p    = Int(PrimNDev[b + (p-1)*Int(NBasis)])   # PrimNDev[b,p]
            l_p    = Int(PrimLDev[b + (p-1)*Int(NBasis)])
            m_p    = Int(PrimMDev[b + (p-1)*Int(NBasis)])
            ζ_p    = PrimζDev[b + (p-1)*Int(NBasis)]
            norm_p = PrimNormDev[b + (p-1)*Int(NBasis)]
            coeffp = PrimCoeffDev[b + (p-1)*Int(NBasis)]
            ynorm  = PrimYNormDev[b + (p-1)*Int(NBasis)]

            # radial factor r^(n-1) * exp(-ζ r)
            radial = 0.0f0
            if r == 0.0f0
                radial = (n_p == 1) ? 1.0f0 : 0.0f0   # r^(n-1) zero if n>1
            else
                radial = r^(n_p - 1) * exp(-ζ_p * r)
            end

            # angular part: compute associated Legendre P_l^|m|(cosθ) via recurrence
            mm = abs(m_p)
            Plm = 0.0f0

            # compute P_m^m (x) = (-1)^m (2m-1)!! (1-x^2)^{m/2}
            if mm == 0
                Pmm = 1.0f0
            else
                # compute double factorial (2m-1)!! as product of odd integers
                df = 1.0f0
                for k in 1:mm
                    df *= (2k - 1)
                end
                # (-1)^m factor
                if (mm % 2) == 1
                    sign = -1.0f0
                else
                    sign = 1.0f0
                end
                Pmm = sign * df * (sinθ^mm)   # (1 - x^2)^(m/2) == sinθ^m
            end

            if l_p == mm
                Plm = Pmm
            elseif l_p == mm + 1
                Plm = cosθ * (2mm + 1) * Pmm
            else
                # use recurrence for l >= m+2
                Pprev = Pmm
                Pcur = cosθ * (2mm + 1) * Pmm  # P_{m+1}^m
                for ll in (mm + 2):l_p
                    # P_l^m = ((2l-1) x P_{l-1}^m - (l + m -1) P_{l-2}^m) / (l - m)
                    numer = (2*ll - 1) * cosθ * Pcur - (ll + mm - 1) * Pprev
                    denom = ll - mm
                    Pnext = numer / denom
                    Pprev, Pcur = Pcur, Pnext
                end
                Plm = Pcur
            end

            # real spherical harmonic value
            realY = 0.0f0
            if m_p == 0
                realY = ynorm * Plm
            elseif m_p > 0
                realY = sqrt(2.0f0) * ynorm * Plm * cos(float(m_p) * φ)
            else # m_p < 0
                realY = sqrt(2.0f0) * ynorm * Plm * sin(float(mm) * φ)
            end

            # primitive contribution
            primVal = norm_p * coeffp * radial * realY
            val_total += primVal
        end

        AODev[idx] = val_total
    end
    return
end

function KernelComputeRho!(RhoDev, AODev, PDev, NBasis, NPoints)

    # GPU kernel to compute rho for each point:
    # For each point j we have AOvals[:,j] on device and Pdev (nbasis x nbasis).
    # Kernel computes rho[j] = a' * P * a by first doing tmp_i = sum_mu P[i,mu]*a[mu]
    # then rho = sum_i a[i]*tmp_i
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if j <= NPoints
        # AODev is stored column-major as (NBasis, NPoints) flattened
        # compute tmp = P * a
        # tmp_idx i from 1:NBasis
        rho_val = zero(eltype(RhoDev))
        @inbounds for i in 1:NBasis
            tmp = zero(eltype(RhoDev))
            ai = AODev[(i-1)*NPoints + j]  # AODev[i,j]
            # multiply row i of PDev with column a
            # PDev is NBasis x NBasis row-major in device memory (Julia column-major still)
            # access PDev[k + (i-1)*NBasis] -> P[i,k]
            @inbounds for k in 1:NBasis
                a_k = AODev[(k-1)*NPoints + j] # AODev[k,j]
                tmp += PDev[k + (i-1)*NBasis] * a_k
            end
            rho_val += ai * tmp
        end
        RhoDev[j] = rho_val
    end
    return
end

# wrapper to launch kernel
function ComputeRhoOnGPU!(AOMatrix::Matrix{Float32}, PMatrix::Matrix{Float64})
    # AOMatrix: NBasis x NPoints (Column-major)
    NBasis, NPoints = size(AOMatrix)

    # Move arrays to device
    AODev = cu(AOMatrix)            # Float32
    # convert P to Float32 on device for speed and compatibility
    PDev = cu(Array{Float32}(PMatrix))  # NBasis x NBasis
    RhoDev = CUDA.zeros(Float32, NPoints)

    threads = 128
    blocks = cld(NPoints, threads)
    @cuda threads=threads blocks=blocks KernelComputeRho!(RhoDev, AODev, PDev, NBasis, NPoints)

    # retrieve results
    return Array(RhoDev)
end