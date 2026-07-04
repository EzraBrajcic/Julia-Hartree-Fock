using LinearAlgebra
using PrettyTables
using TimerOutputs

include("../src/lib/STOFuncs.jl")
include("../src/lib/Reader.jl")
include("../src/lib/CUHFSCF.jl")

to = TimerOutput()

Basis = GenerateBasisSet(ReadAtomBFs("Ne_1S.dat"), 10)

# ── Electronic structure parameters ───────────────────────────────────────
occα = [1,1,1,1,1]
occβ = [1,1,1,1,1]
EC   = GenerateEC(occα, occβ)

# ── Working arrays ───────────────────────────────────────────────────────
BasisSize = Basis.BasisSize
J         = zeros(Float64, BasisSize, BasisSize)
K         = zeros(Float64, BasisSize, BasisSize)
KM        = zeros(Float64, BasisSize, BasisSize)
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
C_InitR = C_Init
C_InitR[4:5, 4:5] *= [cos(π/4) -sin(π/4); sin(π/4) cos(π/4)]
Pα, Pβ, P_Init, M = DensityMatrices(C_Init, C_Init, EC)

# ── SCF procedure ─────────────────────────────────────────────────────────
max_iter = 2000
ConvC    = 1.0e-18
P_Guess  = P_Init

FinalEnergy, Ei, Count, ΔPα, ΔPβ, FNR, max_R,
evecsα, evecsβ, evalsα, evalsβ,
J, K, Pα, Pβ, P, M, S, S_Ortho,
Fcs, Fα, Fβ, Fα_Ortho, Fβ_Ortho,
Cα, Cβ,
to = SCF(Basis, EC, P_Guess, ConvC, max_iter, to)

N = Basis.BasisSize
T = zeros(Float64, N, N)
U = zeros(Float64, N, N)

# Compute kinetic + nuclear potential matrix elements

# Iterating only from μ = 1 to N and ν = μ to N to exploit
# symmetry conditions since Hamiltonian matrix is symmetric about the diagonal.
# Off-diagonal values can then just be copied from the first half of the matrix
# about the diagonal reducing the number of computations by half
for μ in 1:N
    for ν in 1:N
        Tsumμν = 0.0
        Usumμν = 0.0
        for STOμ in Basis.BasisFunctions[μ].STOs
            for STOν in Basis.BasisFunctions[ν].STOs
                Tsumμν += KineticInt(STOμ, STOν)
                Usumμν += NuclearPotentialInt(STOμ, STOν, Basis.z)
            end
        end
        T[μ,ν] = Tsumμν
        U[μ,ν] = Usumμν
    end
end

Ke = tr(P*T)
Pe = tr(P*U)
println("Final total energy (Eₕ) :                    ", FinalEnergy)
println("Total kinetic energy (T):                    ", Ke)
println("Total nuclear potential energy (U):          ", Pe)
println("Virial theorem check Hcore (U/T):            ", Pe/Ke)
println("Virial theorem with Coulomb ((E_F - T)/T):   ", (FinalEnergy - Ke)/Ke)
println("J and K energy (E_F - T - U):                ", FinalEnergy - Ke - Pe)
println("Nuclear potential + Coulomb (E_F - T):       ", FinalEnergy - Ke)