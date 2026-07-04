using LinearAlgebra
using GLMakie
using CUDA

include("lib/CUHFSCF.jl")
include("lib/GPUPlotPoint.jl")
include("lib/Reader.jl")

function PlotElectronDensityVolume(Basis, P; range=(-3.0, 3.0), resolution=50, isovalue=1.0e-2)
    # Creates a 3D volumetric visualization of electron density

    # Arguments:
    # `Basis`: BasisSet object containing the molecular orbital basis
    # `P`: Density matrix from SCF calculation
    # `range`: Tuple specifying the spatial range (min, max) in atomic units
    # `resolution`: Number of grid points along each axis
    # `isovalue`: Density threshold for isosurface rendering
    
    # Generate 3D grid
    coords = LinRange(range[1], range[2], resolution)
    
    X = [x for x in coords, y in coords, z in coords]
    Y = [y for x in coords, y in coords, z in coords]
    Z = [z for x in coords, y in coords, z in coords]
    
    X_flat = vec(X)
    Y_flat = vec(Y)
    Z_flat = vec(Z)
    NPoints = length(X_flat)
    
    println("Computing density on $(resolution)³ = $(NPoints) grid points...")
    
    # Pack primitives for GPU
    MaxPrims, PrimN, PrimL, PrimM, Primζ, PrimNorm, PrimCoeff, 
        PrimYNorm, PrimCount = PackPrimitives(Basis)
    
    NBasis = Basis.BasisSize
    
    # Transfer to GPU
    XDev = cu(X_flat)
    YDev = cu(Y_flat)
    ZDev = cu(Z_flat)
    AODev = CUDA.zeros(Float32, NBasis * NPoints)
    
    PrimNDev = cu(PrimN)
    PrimLDev = cu(PrimL)
    PrimMDev = cu(PrimM)
    PrimζDev = cu(Primζ)
    PrimNormDev = cu(PrimNorm)
    PrimCoeffDev = cu(PrimCoeff)
    PrimYNormDev = cu(PrimYNorm)
    PrimCountDev = cu(PrimCount)
    
    # Compute AO values
    threads = 256
    total = NBasis * NPoints
    blocks = cld(total, threads)
    
    println("Launching GPU kernel with $(blocks) blocks, $(threads) threads...")
    
    @cuda threads=threads blocks=blocks KernelComputeAOWithAng!(
        AODev, XDev, YDev, ZDev,
        PrimNDev, PrimLDev, PrimMDev,
        PrimζDev, PrimNormDev, PrimCoeffDev, PrimYNormDev,
        PrimCountDev,
        Int32(NBasis), Int32(MaxPrims), Int32(NPoints)
    )
    
    # Compute density
    println("Computing electron density...")
    AOMatrix = reshape(Array(AODev), NBasis, NPoints)
    rho_flat = ComputeRhoOnGPU!(AOMatrix, P)
    
    # Reshape to 3D grid
    rho_volume = reshape(rho_flat, resolution, resolution, resolution)
    
    # Calculate voxel volume from actual grid spacing
    dx = coords[2] - coords[1]
    voxel_volume = dx^3
    
    # Multiply density by voxel volume for proper normalization
    rho_volume = rho_volume .* voxel_volume
    
    println("Density range: [$(minimum(rho_volume)), $(maximum(rho_volume))] a.u.")
    println("Voxel volume: $(voxel_volume) a.u.")
    println("Integrated density: $(sum(rho_volume)) electrons")

    # Create 3D visualization with volumetric rendering
    fig = Figure(size=(1440, 1440))
    ax = Axis3(fig[1, 1], 
               xlabel="x (a₀)", 
               ylabel="y (a₀)", 
               zlabel="z (a₀)",
               title="Electron Density Volume",
               aspect=(1, 1, 1))


    
    # Extract endpoints for volume specification
    x_range = (coords[1], coords[end])
    y_range = (coords[1], coords[end])
    z_range = (coords[1], coords[end])
    
    # Define threshold in log space
    rho_log = log10.(rho_volume)
    threshold_log = log10(isovalue)
    vmin, vmax = minimum(threshold_log), maximum(rho_log)

    colormap = to_colormap(:plasma)
    colormap[1] = RGBAf(7,7,7,0)

    vol = volume!(ax, x_range, y_range, z_range, rho_log,
                algorithm = :absorption,
                colormap = colormap,
                colorrange = (vmin, vmax),
                absorption = 3.5,
                shading = true,
                backlight = 0.75,
                specular = 0.5,
                shininess = 8)
    
    Colorbar(fig[1, 2], vol, label="log(ρ(r)) (a.u.)")
    
    
    println("Visualization complete!")
    
    return fig
end

function main()

    to = TimerOutput()

    # ── Basis set construction ────────────────────────────────────────────────
    Basis = GenerateBasisSet(ReadAtomBFs("Ti_3F.dat"), 22)

    # ── Electronic structure parameters ───────────────────────────────────────
    occα = [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]
    occβ = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
    EC   = GenerateEC(occα, occβ)

    # ── Working arrays ───────────────────────────────────────────────────────
    BasisSize = Basis.BasisSize
    J         = zeros(Float64, BasisSize, BasisSize)
    K         = zeros(Float64, BasisSize, BasisSize)
    Fα        = zeros(Float64, BasisSize, BasisSize)
    Fβ        = zeros(Float64, BasisSize, BasisSize)
    Fα_Ortho  = zeros(Float64, BasisSize, BasisSize)
    Fβ_Ortho  = zeros(Float64, BasisSize, BasisSize)
    Cα        = zeros(Float64, BasisSize, BasisSize)
    Cβ        = zeros(Float64, BasisSize, BasisSize)
    Pα        = zeros(Float64, BasisSize, BasisSize)
    Pβ        = zeros(Float64, BasisSize, BasisSize)
    P_Old     = zeros(Float64, BasisSize, BasisSize)
    evecsα    = zeros(Float64, BasisSize, BasisSize)
    evecsβ    = zeros(Float64, BasisSize, BasisSize)
    evalsα    = zeros(Float64, BasisSize)
    evalsβ    = zeros(Float64, BasisSize)

    # ── Integral evaluation ───────────────────────────────────────────────────
    S        = OverlapMatrix(Basis, to)
    S_Ortho  = Orthogonalize(S, to)
    H        = CoreHamiltonian(Basis, to)
    H_ortho  = S_Ortho' * H * S_Ortho
    H_evals, H_evecs = eigen(Symmetric(H_ortho))
    C_Init = S_Ortho * H_evecs

    Cα_InitR = C_Init
    Cβ_InitR = C_Init
    Cβ_InitR[14:15, 14:15] *= [cos(π/4) -sin(π/4); sin(π/4) cos(π/4)]
    Pα, Pβ, P_Init, M = DensityMatrices(Cα_InitR, Cβ_InitR, EC)

    # ── SCF procedure ─────────────────────────────────────────────────────────
    max_iter = 50000
    ConvC    = 5.0e-14
    P_Guess  = P_Init

    FinalEnergy, Ei, Count, ΔPα, ΔPβ, RMS_R, max_R,
    evecsα, evecsβ, evalsα, evalsβ,
    J, K, Pα, Pβ, P, M, S, S_Ortho,
    Fcs, Fα, Fβ, Fα_Ortho, Fβ_Ortho,
    Cα, Cβ = SCF(Basis, EC, P_Guess, ConvC, max_iter, to)

    print_timer(to)

    println("\n── SCF Results ──────────────────────────────────────────────────")
    println("Final Energy  (Eₕ) :      ", FinalEnergy)
    println("Iterations         :      ", Count < max_iter ? string(Count) : "DID NOT CONVERGE")
    println("Final ΔPα          :      ", ΔPα)
    println("Final ΔPβ          :      ", ΔPβ)
    println("Final RMS of Residual:    ", RMS_R)
    println("Final absmax of Residual: ", max_R)

    set_theme!(theme_dark())
    
    fig = PlotElectronDensityVolume(Basis, P; range=(-0.75, 0.75), resolution=500, isovalue=3.75e-8)
    display(fig)
    save("Data/Output/Titanium(3F) 1 Electron Density (tight absorption).png", fig, update=false)
end

# Run main when executed
if !isdefined(Base, :test) 
    main() 
end

