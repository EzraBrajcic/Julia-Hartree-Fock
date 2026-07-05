using GSL

include("DataTypes.jl")

function OverlapInt(STO1, STO2)
    # Computes the 1-centre overlap integral between two STOs, which is defined as S = ∫φ†_1(r)φ_2(r) dr. The integral can be solved analytically using the formula S = δ_ll'δ_mm' * N1 * N2 * c1 * c2 * (n1 + n2)! / (ζ1 + ζ2)^(n1 + n2 + 1) where N is the normalization constant, c is the contraction coefficient, n is the principal quantum number, and ζ is the orbital exponent.
    if STO1.l != STO2.l || STO1.m != STO2.m
        return 0.0
    else
        return STO1.coeff * STO2.coeff * (STO1.normconst * STO2.normconst) * factorial(STO1.n + STO2.n)/((STO1.ζ + STO2.ζ)^(STO1.n + STO2.n + 1))
    end
end

function KineticInt(STO1, STO2)
    # Computes the 1-centre kinetic energy integral between two STOs, which is defined as T = -1/2 ∫φ†_1(r)∇^2φ_2(r) dr. The integral can be solved analytically using the formula T = δ_ll'δ_mm' * 1/2 * N1 * N2 * c1 * c2 * (n1 + n2 - 1)! / (ζ1 + ζ2)^(n1 + n2) * (ζ2^2 - ζ2 * (n1 + n2) + (n1 + n2)(n1 + n2 - 1)) where N is the normalization constant, c is the contraction coefficient, n is the principal quantum number, and ζ is the orbital exponent.
    if STO1.l != STO2.l || STO1.m != STO2.m
        return 0.0
    else
        gammaint1 = ((STO2.l * (STO2.l + 1) - STO2.n * (STO2.n - 1))) * factorial(STO1.n + STO2.n - 2) / (STO1.ζ + STO2.ζ)^(STO1.n + STO2.n - 1)
        gammaint2 = 2 * STO2.ζ * STO2.n * factorial(STO1.n + STO2.n - 1) / (STO1.ζ + STO2.ζ)^(STO1.n + STO2.n)
        gammaint3 = -STO2.ζ^2 * factorial(STO1.n + STO2.n) / (STO1.ζ + STO2.ζ)^(STO1.n + STO2.n + 1)

        return 0.5 * STO1.coeff * STO2.coeff * (STO1.normconst * STO2.normconst) * (gammaint1 + gammaint2 + gammaint3)
    end
end

function NuclearPotentialInt(STO1, STO2, Z::Int)
    # Computes the 1-centre nuclear potential integral between two STOs, which is defined as V = -Z ∫φ†_1(r)1/rφ_2(r) dr. The integral can be solved analytically using the formula V = δ_ll'δ_mm' * N1 * N2 * c1 * c2 * -Z(n1 + n2 - 1)! / (ζ1 + ζ2)^(n1 + n2) where N is the normalization constant, c is the contraction coefficient, n is the principal quantum number, ζ is the orbital exponent, and Z is the nuclear charge of the atom that the STO is attached to.
    if STO1.l != STO2.l || STO1.m != STO2.m
        return 0.0
    else
        return STO1.coeff * STO2.coeff * (STO1.normconst * STO2.normconst) * -Z * factorial(STO1.n + STO2.n - 1) / (STO1.ζ + STO2.ζ)^(STO1.n + STO2.n)
    end
end

function CoulombInt(a::STO, b::STO, c::STO, d::STO)
    # Represents the integral <φ_a φ_b|1/|r1-r2||φ_c φ_d> = ∫∫φ†_a(r1)φ_c(r1)1/|r1-r2|φ†_b(r2)φ_d(r2) dr1dr2 (also represented as (ab|cd)) such that the radial coordinates follow the form of (12|12). The 1/|r1-r2| term can be transformed into a Laplace expansion of the form ∑_(L=0)^(∞) (4pi/(2L + 1)) * ∑_(M=-L)^(L) r_<^L/r_>^(L+1) * Y_Lω(Ω1)Y*_Lω(Ω2) where r_< is the smaller of r1 and r2 and r_> is the larger of r1 and r2. The angular portion of the integral can then be solved using Gaunt coefficients which are products of 3j-Wigner symbols, while the radial portion can be solved using incomplete gamma functions.
    result::Float64 = 0.0

    # Bounds of L follow from selection rules for product of 2 3j-Wigner symbols. While technically 2 Gaunt coefficient products of (l_a, l_c, L, 0, 0, 0)(l_a, l_b, L, -m_a, m_c, ω) and (l_b, l_d, L, 0, 0, 0)(l_b, l_d, L, -m_b, m_d, ω), because the upper portion of the 3j-Wigner symbols in each product are identical, they follow the same j_i-dependent selection rules. This allows us to only consider the selection rules of the (l_a, l_b, L, -m_a, m_c, ω) and (l_b, l_d, L, -m_b, m_d, ω) 3j-Wigner symbols.
    
    # Any L indice in the series that doesn't obey the selection rules automatically sets the sum term to 0, truncating infinite sum of the Laplace expansion to a finite number of terms.
    L_min::Int = max(abs(a.l - c.l), abs(b.l - d.l)) 
    L_max::Int = min(a.l + c.l, b.l + d.l)

    acn_Sum::Int = a.n + c.n
    bdn_Sum::Int = b.n + d.n

    ζ::Float64 = a.ζ + c.ζ + b.ζ + d.ζ

    for L in L_min:L_max
        M_Sum::Float64 = 0.0
        L_Sum::Float64 = 0.0

        for M in -L:L
            GauntPrefactor::Float64 = 0.0
            ac_GauntCoefficient::Float64 = 0.0
            bd_GauntCoefficient::Float64 = 0.0

            # More selection rules for Coulomb integral Gaunt coefficients
            if ((a.m == 0 && c.m == 0 && M == 0) && ((a.l + c.l + L) % 2) != 0) || ((b.m == 0 && d.m == 0 && M == 0) && ((b.l + d.l + L) % 2 != 0))
                M_Sum += 0.0

            elseif ((M == c.m - a.m) && (M == b.m - d.m))
                GauntPrefactor = sqrt((2a.l + 1) * (2c.l + 1) * (2b.l + 1) * (2d.l + 1)) * (2L + 1) / (4pi)

                ac_GauntCoefficient = GSL.sf_coupling_3j(2*a.l, 2*c.l, 2*L, 0, 0, 0) * GSL.sf_coupling_3j(2*a.l, 2*c.l, 2*L, -2*a.m, 2*c.m, -2*M)
                bd_GauntCoefficient = GSL.sf_coupling_3j(2*b.l, 2*d.l, 2*L, 0, 0, 0) * GSL.sf_coupling_3j(2*b.l, 2*d.l, 2*L, -2*b.m, 2*d.m, 2*M)
                
                Phase::Float64 = isodd(b.m + c.m) ? -1.0 : 1.0

                M_Sum += Phase * GauntPrefactor * ac_GauntCoefficient * bd_GauntCoefficient
            else
                M_Sum += 0.0
            end

        end
        if M_Sum == 0.0
            L_Sum = 0.0

        else
            # Calculate radial integral portion

            #=
            The Laplace expansion breaks the radial integral ∫∫R†_a(r1)R†_b(r2)r_<^L/r_>^(L+1)R_c(r1)R_d(r2)r1^2 * r2^2 dr1dr2 into 
            
            ∫R†_a(r1)R_c(1)r1^(1-L) dr1 * (∫R†_b(r2)R_d(r2)r^(L+2) dr2) on the region r2 = 0 to r2 = r1 where r2 < r1 and 
            
            ∫R†_a(r1)R_c(1)r1^(2+L) dr1 * (∫R†_b(r2)R_d(r2)r^(1-L) dr2) on the region r2 = r1 to infinity where r2 > r1.
            
            This gives us 2 integrals of the form:

            ∫R†_a(r1)R_c(r1)r1^(1-L) * γ(n_b + n_d + L + 1, (ζ_b + ζ_d)r1) dr1 + 
            ∫R†_a(r1)R_c(r1)r1^(2+L) * Γ(n_b + n_d - L, (ζ_b + ζ_d)r1) dr1

            where γ(s, x) is the lower incomplete gamma function and Γ(s, x) is the upper incomplete gamma function. The gamma functions can be expanded into finite sums of exponentials and powers since their first arguments are quantum numbers which are integers. The summation can then be taken outside of the integral, leaving integrals of the form ∫r^n * e^(-ζr) dr which can be solved analytically.
            =#

            L_Sum = 0.0

            Γr1::Float64 = 0.0
            Γr1_Prefactor::Float64 = factorial(b.n + d.n - L - 1)/(b.ζ + d.ζ)^(bdn_Sum - L)

            for i in 0:(bdn_Sum - L - 1)
                Γr1 += factorial(acn_Sum + L + i)  * (b.ζ + d.ζ)^(i) / (factorial(i) * (ζ)^(acn_Sum + L + i + 1))
            end
            Γr1 *= Γr1_Prefactor

            Γr2::Float64 = 0.0
            Γr2_Prefactor::Float64 = factorial(a.n + c.n - L - 1)/(a.ζ + c.ζ)^(acn_Sum - L)

            for i in 0:(acn_Sum - L - 1)
                Γr2 += factorial(bdn_Sum + L + i)  * (a.ζ + c.ζ)^(i) / (factorial(i) * (ζ)^(bdn_Sum + L + i + 1))
            end
            Γr2 *= Γr2_Prefactor  

            L_Sum += Γr1 + Γr2
            
            result += 4pi/(2L + 1) * L_Sum * M_Sum
        end
    end

    # Multiply by normalization and contraction coefficients for all 4 STOs
    return a.coeff * b.coeff * c.coeff * d.coeff * (a.normconst * b.normconst * c.normconst * d.normconst) * result
end

function ExchangeInt(a::STO, b::STO, c::STO, d::STO)
    
    # Exact same form as Coulomb integral, just swapping indices b c and then b d, changing from coordinates of (12|12) to (12|21)
    return CoulombInt(a, c, d, b)
end

