using GLMakie
using Printf
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
#  Colour palette
# ─────────────────────────────────────────────────────────────────────────────
BG        = RGBf(0.08, 0.09, 0.11)
PANEL_BG  = RGBf(0.11, 0.13, 0.16)
BORDER    = RGBf(0.22, 0.26, 0.32)
FG        = RGBf(0.88, 0.90, 0.93)
FG_DIM    = RGBf(0.50, 0.55, 0.62)
ACCENT    = RGBf(0.35, 0.75, 0.95)    # α / cool blue
ACCENT2   = RGBf(0.95, 0.65, 0.25)    # β / amber
GOOD      = RGBf(0.35, 0.85, 0.55)    # converged green
BAD       = RGBf(0.95, 0.35, 0.35)    # not converged red
ROW_ALT   = RGBf(0.13, 0.15, 0.19)    # table alternating row

CMAP_DIV  = :seaborn_icefire_gradient     # diverging – zero at white; paired coefficient/Fock matrices
CMAP_DENS = :seaborn_mako_gradient  # sequential – density matrices (non-negative)
CMAP_SPIN = :seaborn_mako_gradient     # diverging – spin density (signed, zero = no spin)


# ─────────────────────────────────────────────────────────────────────────────
#  Colour-limit helpers
# ─────────────────────────────────────────────────────────────────────────────
function paired_clim(A, B)
    m = max(abs(minimum(A)), abs(maximum(A)), abs(minimum(B)), abs(maximum(B)))
    return (-m, m)
end

function sym_clim(A)
    m = max(abs(minimum(A)), abs(maximum(A)))
    return (-m, m)
end

# For non-negative matrices that should share a common scale.
function pos_clim(mats...)
    lo = minimum(minimum(m) for m in mats)
    hi = maximum(maximum(m) for m in mats)
    return (lo, hi)
end

function SpinContamination(Pα, Pβ, EC::ElectronConfig)

    # ─────────────────────────────────────────────────────────────────────────────
    #  SpinContamination – compute ⟨S²⟩ for a UHF wavefunction.
    #  Returns (S2_exact, S2_uhf, contamination = S2_uhf - S2_exact)
    # ─────────────────────────────────────────────────────────────────────────────

    S_val  = (EC.Nα - EC.Nβ) / 2.0
    S2_exact = S_val * (S_val + 1.0)

    δs = EC.Nβ - tr(Pα * Pβ)

    S2_uhf = S2_exact + δs
    return S2_exact, S2_uhf, δs
end

function MatrixAxis!(ax, M; Title="", cmap=CMAP_DIV, clim=sym_clim(M), mat_name="M")

    # ─────────────────────────────────────────────────────────────────────────────
    #  MatrixAxis! – heatmap with cell-grid lines and full-precision hover tooltip.
    #
    #  GLMakie's DataInspector reads hm.inspector_label(plot, idx, pos) to build
    #  the tooltip string.  We attach a closure that reports M[i,j] with full
    #  Float64 precision.  inspectable must be true (default here).
    # ─────────────────────────────────────────────────────────────────────────────

    n = size(M, 1)

    hm = heatmap!(ax, 1:n, 1:n, M;
        colormap    = cmap,
        colorrange  = clim,
        inspectable = true
    )

    hm.inspector_label = (plot, idx, pos) -> begin
        i = clamp(round(Int, pos[2]), 1, n)   # y → row
        j = clamp(round(Int, pos[1]), 1, n)   # x → col
        @sprintf("%s[%d, %d] = %.15g", mat_name, i, j, M[j, i])
    end

    for k in 0.5:1:(n + 0.5)
        hlines!(ax, k; color = (BORDER, 0.45), linewidth = 0.6)
        vlines!(ax, k; color = (BORDER, 0.45), linewidth = 0.6)
    end

    ax.title             = Title
    ax.titlecolor        = FG
    ax.titlesize         = 13
    ax.xticklabelsize    = 10
    ax.yticklabelsize    = 10
    ax.xticklabelcolor   = FG_DIM
    ax.yticklabelcolor   = FG_DIM
    ax.xticks            = 1:n
    ax.yticks            = 1:n
    ax.yreversed         = true
    ax.backgroundcolor   = PANEL_BG
    ax.topspinecolor     = BORDER
    ax.bottomspinecolor  = BORDER
    ax.leftspinecolor    = BORDER
    ax.rightspinecolor   = BORDER
    ax.xgridvisible      = false
    ax.ygridvisible      = false
    return hm
end

function StyleAxis!(ax; Title="", xlabel="", ylabel="")

    # ─────────────────────────────────────────────────────────────────────────────
    #  StyleAxis! – shared theme for non-heatmap axes
    # ─────────────────────────────────────────────────────────────────────────────

    ax.backgroundcolor   = PANEL_BG
    ax.topspinecolor     = BORDER
    ax.bottomspinecolor  = BORDER
    ax.leftspinecolor    = BORDER
    ax.rightspinecolor   = BORDER
    ax.xgridvisible      = false
    ax.ygridcolor        = (BORDER, 0.6)
    ax.ygridwidth        = 0.8
    ax.Title             = Title
    ax.Titlecolor        = FG
    ax.Titlesize         = 13
    ax.xlabel            = xlabel
    ax.xlabelcolor       = FG_DIM
    ax.ylabel            = ylabel
    ax.ylabelcolor       = FG_DIM
    ax.xticklabelcolor   = FG_DIM
    ax.yticklabelcolor   = FG_DIM
    ax.xticklabelsize    = 10
    ax.yticklabelsize    = 10
end

function SectionLabel!(fig, row, cols, text_str)

    # ─────────────────────────────────────────────────────────────────────────────
    #  SectionLabel! – small section header in its own fixed-height grid row.
    #  Using a dedicated row (not Top()) prevents any overlap with plot content.
    # ─────────────────────────────────────────────────────────────────────────────

    Label(fig[row, cols],
        text_str;
        fontsize  = 9,
        color     = FG_DIM,
        halign    = :left,
        valign    = :bottom,
        tellwidth = false,
        padding   = (4, 0, 0, 2)
    )
end

function EigenvalueTable!(ax, evalsα, evalsβ, occα, occβ)

    # ─────────────────────────────────────────────────────────────────────────────
    #  EigenvalueTable! – α/β eigenvalue table with occupation flags.
    #
    #  Column layout (normalised x in [0,1]):
    #    0.04  │  #
    #    0.12  │  εα (Eₕ)          ← 36 % of width
    #    0.50  │  εβ (Eₕ)          ← 36 % of width
    #    0.97  │  occ (right-align) ← remaining, no clipping
    # ─────────────────────────────────────────────────────────────────────────────

    n  = length(evalsα)
    rh = 1.0 / (n + 1)      # +1 for header row

    hidedecorations!(ax)

    # x anchor positions
    x_idx  = 0.04
    x_ea   = 0.14
    x_eb   = 0.52
    x_occ  = 0.97            # right-aligned

    # Header row
    y_top = 1.0;  y_bot = y_top - rh;  y_mid = (y_top + y_bot) / 2
    poly!(ax, Point2f[(0,y_bot),(1,y_bot),(1,y_top),(0,y_top)];
          color = RGBf(0.16, 0.19, 0.24), strokewidth = 0)
    text!(ax, x_idx, y_mid; text = "#",       color = FG_DIM, fontsize = 11,
          align = (:left, :center), font = :bold)
    text!(ax, x_ea,  y_mid; text = "εα (Eₕ)", color = ACCENT,  fontsize = 11,
          align = (:left, :center), font = :bold)
    text!(ax, x_eb,  y_mid; text = "εβ (Eₕ)", color = ACCENT2, fontsize = 11,
          align = (:left, :center), font = :bold)
    text!(ax, x_occ, y_mid; text = "occ",     color = FG_DIM, fontsize = 11,
          align = (:right, :center), font = :bold)
    lines!(ax, [0.0, 1.0], [y_bot, y_bot]; color = BORDER, linewidth = 1.0)

    for i in 1:n
        y_top = 1.0 - i * rh;  y_bot = y_top - rh;  y_mid = (y_top + y_bot) / 2

        poly!(ax, Point2f[(0,y_bot),(1,y_bot),(1,y_top),(0,y_top)];
              color = iseven(i) ? ROW_ALT : PANEL_BG, strokewidth = 0)

        occ_a   = i <= length(occα) ? occα[i] : 0
        occ_b   = i <= length(occβ) ? occβ[i] : 0
        occ_str = "$(occ_a)α $(occ_b)β"
        occ_col = (occ_a + occ_b) > 0 ? GOOD : FG_DIM

        text!(ax, x_idx, y_mid; text = string(i),
              color = FG_DIM, fontsize = 11, align = (:left, :center))
        text!(ax, x_ea,  y_mid; text = @sprintf("%.7f", evalsα[i]),
              color = ACCENT,  fontsize = 11, align = (:left, :center), font = :bold)
        text!(ax, x_eb,  y_mid; text = @sprintf("%.7f", evalsβ[i]),
              color = ACCENT2, fontsize = 11, align = (:left, :center), font = :bold)
        text!(ax, x_occ, y_mid; text = occ_str,
              color = occ_col, fontsize = 11, align = (:right, :center))

        lines!(ax, [0.0, 1.0], [y_bot, y_bot]; color = BORDER, linewidth = 0.6)
    end

    xlims!(ax, 0, 1);  ylims!(ax, 0, 1)
end

function shared_colorbar!(fig, pos, cmap, clim, lbl)

    # ─────────────────────────────────────────────────────────────────────────────
    #  shared_colorbar! – thin vertical colorbar, factored for reuse
    # ─────────────────────────────────────────────────────────────────────────────

    Colorbar(fig[pos...];
        colormap       = cmap,
        limits         = clim,
        label          = lbl,
        labelcolor     = FG_DIM,
        ticklabelcolor = FG_DIM,
        ticklabelsize  = 9,
        labelsize      = 10,
        width          = 12,
        ticksize       = 4)
end

function PlotResultsSCF(
        Basis, EC,
        H_evecs, C_Init, P_Init,
        ConvC, Count, max_iter,
        FinalEnergy, ΔPα, ΔPβ,
        Ei,
        evalsα, evalsβ,
        evecsα, evecsβ,
        Cα, Cβ,
        Fcs, Fα, Fβ, Fα_Ortho, Fβ_Ortho,
        Pα, Pβ, P, M;
        output_file = "scf_results.png"
    )

    # ─────────────────────────────────────────────────────────────────────────────
    #  PlotResultsSCF  – main entry point
    #
    #    PlotResultsSCF(
    #        Basis, EC,
    #        H_evecs, C_Init, P_Init,
    #        ConvC, Count, max_iter,
    #        FinalEnergy, ΔPα, ΔPβ,
    #        Ei,
    #        evalsα, evalsβ,
    #        evecsα, evecsβ,
    #        Cα, Cβ,
    #        Fcs, Fα, Fβ, Fα_Ortho, Fβ_Ortho,
    #        Pα, Pβ, P, M;     
    #        output_file = "scf_results.png"
    #    )
    #
    #  Grid layout  (row 0 = Title banner)
    #  ┌────────┬─────────────────────────────────────────────┬──────┐
    #  │        │  section label row  (Fixed 16)              │      │
    #  │  Left  │─────────────────────────────────────────────│  CB  │
    #  │  panel │  matrix row  (Relative)                     │      │
    #  │        │─────────────────────────────────────────────│      │
    #  │ (info  │  ... × 4 matrix-row pairs ...               │      │
    #  │ table) ├─────────────────────────────────────────────┤      │
    #  │        │  section label (conv) | eigenvalue table    │      │
    #  │ + eval │─────────────────────────────────────────────│      │
    #  │  table │  convergence plot     | eigenvalue table    │      │
    #  └────────┴─────────────────────────────────────────────┴──────┘
    #
    #  Matrix rows:
    #   A  Fcs        | Cα          | Cβ          | H_evecs (core-H)
    #   B  Fα         | Fβ          | Fα_ortho    | Fβ_ortho
    #   C  evecsα     | evecsβ      | C_Init      | M  (spin density Pα−Pβ)
    #   D  Pα (final) | Pβ (final)  | P (initial) | P  (final total)
    # ─────────────────────────────────────────────────────────────────────────────

    converged = Count < max_iter
    Nelec     = EC.Nα + EC.Nβ
    n         = Basis.BasisSize

    # Spin contamination
    S2_exact, S2_uhf, S2_contam = SpinContamination(Pα, Pβ, EC)

    # ── Figure ────────────────────────────────────────────────────────────────
    # 6 columns  (col 1 = left panel, cols 2-5 = matrices, col 6 = colorbars)
    # 11 rows:
    #   0          Title banner                        Fixed(38)
    #   1          section label A                     Fixed(16)
    #   2          matrix row A                        Relative
    #   3          section label B                     Fixed(16)
    #   4          matrix row B                        Relative
    #   5          section label C                     Fixed(16)
    #   6          matrix row C                        Relative
    #   7          section label D                     Fixed(16)
    #   8          matrix row D  (density)             Relative
    #   9          section label conv/evals            Fixed(16)
    #  10          convergence plot (cols 2-5)
    #              eigenvalue table (col 1)            Relative
    # ─────────────────────────────────────────────────────────────────────────
    fig = Figure(
        resolution      = (2560, 2000),
        backgroundcolor = BG,
        figure_padding  = (18, 18, 18, 18)
    )

    # ── Title banner ──────────────────────────────────────────────────────────
    Box(fig[0, 1:6]; color = (BORDER, 0.20), strokewidth = 0)
    Label(fig[0, 1:6], "UHF-SCF  Dashboard";
          fontsize = 24, font = :bold, color = FG,
          halign = :left, tellwidth = false, padding = (16, 0, 0, 0))
    Label(fig[0, 3:6], "Unrestricted Hartree–Fock  ·  STO Basis";
          fontsize = 13, color = FG_DIM,
          halign = :right, tellwidth = false, padding = (0, 16, 0, 0))

    # ═══════════════════════════════════════════════════════════════════════════
    #  LEFT PANEL (rows 1-8) – SCF info table
    # ═══════════════════════════════════════════════════════════════════════════
    occ_fmt(occ) = join(string.(occ), "  ")

    table_rows = [
        ("Multiplicity",    string(EC.Multiplicity)),
        ("Nα  /  Nβ",       "$(EC.Nα)  /  $(EC.Nβ)"),
        ("N electrons",     string(Nelec)),
        ("Basis size",      string(n)),
        ("Occ α",           occ_fmt(EC.occα)),
        ("Occ β",           occ_fmt(EC.occβ)),
        ("Conv criterion",  @sprintf("%.2e", ConvC)),
        ("Max iterations",  string(max_iter)),
        ("Iterations",      string(Count)),
        ("Converged",       converged ? "Yes  ✓" : "No  ✗"),
        ("ΔPα  (final)",    @sprintf("%.4e", ΔPα)),
        ("ΔPβ  (final)",    @sprintf("%.4e", ΔPβ)),
        ("⟨S²⟩ exact",      @sprintf("%.6f", S2_exact)),
        ("⟨S²⟩ UHF",        @sprintf("%.6f", S2_uhf)),
        ("Spin contam.",    @sprintf("%.6f", S2_contam)),
        ("Energy  (Eₕ)",    @sprintf("%.10f", FinalEnergy)),
    ]

    info_ax = Axis(fig[1:8, 1];
        backgroundcolor  = PANEL_BG,
        leftspinecolor   = BORDER, rightspinecolor  = BORDER,
        topspinecolor    = BORDER, bottomspinecolor = BORDER,
        xgridvisible     = false,  ygridvisible     = false,
    )
    hidedecorations!(info_ax)

    nr = length(table_rows)
    rh = 1.0 / nr
    for (i, (lbl, val)) in enumerate(table_rows)
        y_top = 1.0 - (i - 1) * rh
        y_bot = y_top - rh
        y_mid = (y_top + y_bot) / 2

        poly!(info_ax,
            Point2f[(0, y_bot), (1, y_bot), (1, y_top), (0, y_top)];
            color = iseven(i) ? ROW_ALT : PANEL_BG, strokewidth = 0)

        # Colour-code special rows
        vcol = if lbl == "Converged"
                   converged ? GOOD : BAD
               elseif lbl == "Energy  (Eₕ)"
                   ACCENT
               elseif lbl == "⟨S²⟩ UHF"
                   ACCENT2
               elseif lbl == "Spin Contam."
                   abs(S2_contam) < 0.05 ? GOOD : (abs(S2_contam) < 0.15 ? ACCENT2 : BAD)
               else
                   FG
               end

        text!(info_ax, 0.04, y_mid;
              text = lbl, color = FG_DIM, fontsize = 12, align = (:left, :center))
        text!(info_ax, 0.97, y_mid;
              text = val, color = vcol, fontsize = 12, font = :bold,
              align = (:right, :center))
        lines!(info_ax, [0.0, 1.0], [y_bot, y_bot]; color = BORDER, linewidth = 0.7)
    end
    xlims!(info_ax, 0, 1);  ylims!(info_ax, 0, 1)

    # ═══════════════════════════════════════════════════════════════════════════
    #  LEFT PANEL (row 10) – Orbital eigenvalue table
    # ═══════════════════════════════════════════════════════════════════════════
    SectionLabel!(fig, 9, 1, "ORBITAL EIGENVALUES")

    eval_ax = Axis(fig[10, 1];
        backgroundcolor  = PANEL_BG,
        leftspinecolor   = BORDER, rightspinecolor  = BORDER,
        topspinecolor    = BORDER, bottomspinecolor = BORDER,
    )
    EigenvalueTable!(eval_ax, evalsα, evalsβ, EC.occα, EC.occβ)

    # ═══════════════════════════════════════════════════════════════════════════
    #  MATRIX GRID  (cols 2-5, rows 2/4/6/8)
    # ═══════════════════════════════════════════════════════════════════════════

    # Paired colour limits
    clim_C    = paired_clim(Cα, Cβ)
    clim_F    = paired_clim(Fα, Fβ)
    clim_FO   = paired_clim(Fα_Ortho, Fβ_Ortho)
    clim_evec = paired_clim(evecsα, evecsβ)
    clim_Fcs  = sym_clim(Fcs)
    clim_Hev  = sym_clim(H_evecs)
    clim_M    = sym_clim(M)
    clim_Cini = sym_clim(C_Init)
    # All four density matrices (Pα, Pβ, P_Init, P) share one scale
    clim_P    = pos_clim(Pα, Pβ, P_Init, P)

    # ── Row A : Fcs | Cα | Cβ | H_evecs ──────────────────────────────────────
    SectionLabel!(fig, 1, 2:5, "FOCK CORE  ·  COEFFICIENTS  ·  CORE-H EIGENVECTORS")

    axA = [Axis(fig[2, c]; backgroundcolor = PANEL_BG) for c in 2:5]
    MatrixAxis!(axA[1], Fcs;     Title = "Fcs",      cmap = CMAP_DIV,  clim = clim_Fcs,  mat_name = "Fcs")
    MatrixAxis!(axA[2], Cα;      Title = "Cα",                    cmap = CMAP_DIV,  clim = clim_C,    mat_name = "Cα")
    MatrixAxis!(axA[3], Cβ;      Title = "Cβ",                    cmap = CMAP_DIV,  clim = clim_C,    mat_name = "Cβ")
    MatrixAxis!(axA[4], H_evecs; Title = "H Core Eigenvectors",   cmap = CMAP_DIV,  clim = clim_Hev,  mat_name = "H_evecs")
    shared_colorbar!(fig, [2, 6], CMAP_DIV, clim_C, "Cα / Cβ")

    # ── Row B : Fα | Fβ | Fα_ortho | Fβ_ortho ────────────────────────────────
    SectionLabel!(fig, 3, 2:5, "FOCK MATRICES  ·  AO AND ORTHOGONAL BASIS")

    axB = [Axis(fig[4, c]; backgroundcolor = PANEL_BG) for c in 2:5]
    MatrixAxis!(axB[1], Fα;       Title = "Fα",               cmap = CMAP_DIV, clim = clim_F,  mat_name = "Fα")
    MatrixAxis!(axB[2], Fβ;       Title = "Fβ",               cmap = CMAP_DIV, clim = clim_F,  mat_name = "Fβ")
    MatrixAxis!(axB[3], Fα_Ortho; Title = "Fα  (Orthogonal)", cmap = CMAP_DIV, clim = clim_FO, mat_name = "Fα_ortho")
    MatrixAxis!(axB[4], Fβ_Ortho; Title = "Fβ  (Orthogonal)", cmap = CMAP_DIV, clim = clim_FO, mat_name = "Fβ_ortho")
    shared_colorbar!(fig, [4, 6], CMAP_DIV, clim_F, "Fα / Fβ")

    # ── Row C : evecsα | evecsβ | C_Init | M ────────────────────────────
    SectionLabel!(fig, 5, 2:5, "EIGENVECTORS  ·  INITIAL COEFFICIENT MATRIX  ·  SPIN DENSITY")

    axC = [Axis(fig[6, c]; backgroundcolor = PANEL_BG) for c in 2:5]
    MatrixAxis!(axC[1], evecsα;  Title = "Eigenvectors α",     cmap = CMAP_DIV,  clim = clim_evec, mat_name = "evecsα")
    MatrixAxis!(axC[2], evecsβ;  Title = "Eigenvectors β",     cmap = CMAP_DIV,  clim = clim_evec, mat_name = "evecsβ")
    MatrixAxis!(axC[3], C_Init;  Title = "C  (Initial guess)", cmap = CMAP_DIV,  clim = clim_Cini, mat_name = "C_i")
    MatrixAxis!(axC[4], M;  Title = "M",  cmap = CMAP_SPIN, clim = clim_M,    mat_name = "M")
    shared_colorbar!(fig, [6, 6], CMAP_DIV, clim_evec, "evecs α / β")

    # ── Row D : Pα | Pβ | P_Init | P_f (all on shared density scale) ─────
    SectionLabel!(fig, 7, 2:5, "DENSITY MATRICES  ·  SPIN COMPONENTS AND TOTAL")

    axD = [Axis(fig[8, c]; backgroundcolor = PANEL_BG) for c in 2:5]
    MatrixAxis!(axD[1], Pα; Title = "Pα",      cmap = CMAP_DENS, clim = clim_P, mat_name = "Pα")
    MatrixAxis!(axD[2], Pβ; Title = "Pβ",      cmap = CMAP_DENS, clim = clim_P, mat_name = "Pβ")
    MatrixAxis!(axD[3], P_Init;   Title = "P  (Initial)",      cmap = CMAP_DENS, clim = clim_P, mat_name = "P_i")
    MatrixAxis!(axD[4], P;  Title = "P  (Final)",  cmap = CMAP_DENS, clim = clim_P, mat_name = "P_f")
    shared_colorbar!(fig, [8, 6], CMAP_DENS, clim_P, "Density")

    # ═══════════════════════════════════════════════════════════════════════════
    #  CONVERGENCE PLOT  row 10, cols 2-5  (shares row with eigenvalue table)
    # ═══════════════════════════════════════════════════════════════════════════
    SectionLabel!(fig, 9, 2:5, "SCF ENERGY CONVERGENCE")

    conv_ax = Axis(fig[10, 2:5])
    StyleAxis!(conv_ax; xlabel = "Iteration #", ylabel = "E (Eₕ)")

    scatter!(conv_ax, 1:length(Ei), Ei; color = ACCENT, markersize = 8)

    # ── Layout sizing ─────────────────────────────────────────────────────────
    colsize!(fig.layout, 1, Fixed(270))
    for c in 2:5
        colsize!(fig.layout, c, Relative(0.163))
    end
    colsize!(fig.layout, 6, Fixed(55))

    rowsize!(fig.layout,  0, Fixed(38))       # Title
    rowsize!(fig.layout,  1, Fixed(16))       # label A
    rowsize!(fig.layout,  2, Relative(0.15))  # matrix A
    rowsize!(fig.layout,  3, Fixed(16))       # label B
    rowsize!(fig.layout,  4, Relative(0.15))  # matrix B
    rowsize!(fig.layout,  5, Fixed(16))       # label C
    rowsize!(fig.layout,  6, Relative(0.15))  # matrix C
    rowsize!(fig.layout,  7, Fixed(16))       # label D
    rowsize!(fig.layout,  8, Relative(0.15))  # matrix D (density)
    rowsize!(fig.layout,  9, Fixed(16))       # label conv / evals
    rowsize!(fig.layout, 10, Relative(0.15))  # convergence + evals

    # ── Hover inspector ───────────────────────────────────────────────────────
    DataInspector(fig;
        indicator_color  = (ACCENT, 0.8),
        textcolor       = FG,
        backgroundcolor = RGBAf(0.08, 0.09, 0.11, 0.92),
        outline_color    = BORDER,
        fontsize         = 12
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    save(output_file, fig; px_per_unit = 2)
    println("SCF dashboard saved → ", output_file)
    return fig
end