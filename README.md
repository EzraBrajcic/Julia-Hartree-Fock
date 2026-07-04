# Julia Hartree-Fock

> A self-consistent field solver for Hartree-Fock quantum chemistry calculations implemented in Julia.

---

## Overview

This project provides a foundational implementation of the Hartree-Fock self-consistent field method for atomic and molecular systems. The current release focuses on core computational functionality with basic visualization capabilities. Additional features and improvements are under active development.

---

## Current Features

The solver implements the Hartree-Fock self-consistent field algorithm with support for Slater-type orbital basis functions. Electron density visualization is available through GPU-accelerated three-dimensional plotting using CUDA and GLMakie. Basis sets must be manually configured in the source files for each calculation.

---

## Requirements

- **Windows**: 10/11
- **Julia**: 1.9 or later
- **NVIDIA GPU** (optional): For plotting electron density, requires CUDA 13.0+
- **Visual C++ Redistributable**: 2015+ (Windows only)

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Julia Hartree-Fock"
   ```

2. **Run the setup script**:
   
   **Windows**:
   ```cmd
   .\setup.bat
   ```
   
   **macOS/Linux**:
   ```bash
   bash setup.sh
   ```

---

## Usage

Calculations are performed by running the main solver script. Execute `julia --project=. Main.jl` from the project directory. To generate electron density visualizations, run `julia --project=. PlotRho.jl`.

Basis sets are configured directly in the source files. Open Main.jl or PlotRho.jl and locate the basis set definitions near the beginning of the main function. Modify the STO parameters according to your molecular system requirements.

---

## Project Structure

```
Julia Hartree-Fock/
├── src/                            # Source code
│   ├── main.jl                     # Main executable
│   ├── PlotRho.jl                  # Visualization module
│   ├── PlotSCF.jl                  # Information dashboard from SCF results
│   └── lib/
│       ├── DataTypes.jl            # Structs used throughout code
│       ├── GPUPlotPoint.jl         # GPU Accelerated 1-electron density plotting
│       ├── HFSCF.jl                # Hartree-Fock SCF solver
│       ├── CUHFSCF.jl              # ROHF SCF solver
│       ├── Reader.jl               # Reads input data and converts into structs
│       └── STOFuncs.jl             # STO Integrals
├── data/                           # Data files and outputs
│   ├─── Input/
│   │    ├──── Basis Functions/     # Element basis function file location
│   │    └──── Molecule.dat         # Molecule element and position data
│   ├─── Integrals/                 # Fock matrix integral data storage
│   └─── Output/                    # Plot save location
├── Project.toml                    # Julia project manifest
├── Manifest.toml                   # Dependency lock file (auto-generated)
├── setup.bat                       # Windows setup script
├── setup.jl                        # Julia setup script
└── README.md                       # This file
```

---

## Current Limitations

This release represents foundational functionality with several limitations that will be addressed in future versions. Understanding these limitations will help set appropriate expectations for using the software in its current state.

To effectively run the program for a given system, the main.jl file must be edited by the user to create any many-body system. GPU acceleration is only utilized for visualization rendering, not for the core SCF calculations themselves. The computational kernels for matrix operations and integral evaluations run on the CPU regardless of GPU availability.

The implementation currently only supports atomic ROHF calculations for open-shell systems. The code currently assumes single-atom systems centered at the origin, and multi-atom molecular calculations require manual geometry specification.

---

## Troubleshooting

If you encounter CUDA version mismatch errors, remove the Manifest.toml file and rerun the setup script. For dependency conflicts, regenerate the environment by running `julia --project=.` followed by `using Pkg; Pkg.resolve()` in the Julia REPL.

---

## Development Status

This project is under active development with several significant enhancements planned for future releases. The development roadmap includes automated basis set loading from standard quantum chemistry formats such as Gaussian basis set files, which will eliminate the need for manual basis set table creation. Extended GPU acceleration for SCF iterations is planned, which will move computationally intensive matrix operations and integral evaluations to the GPU for systems with NVIDIA hardware.

Improved visualization options for molecular orbitals are planned, including isosurface rendering and orbital energy level diagrams. Multi-atom molecular geometry support with automatic nuclear repulsion calculations will enable routine molecular calculations without manual geometry specification.

---

## License

This project is licensed under the MIT License. Complete license terms are available in the `LICENSE` file included in the repository.

### Important Licensing Considerations

The original code in this repository is released under the MIT License, which provides broad permissions for use and modification. However, this software depends on the GNU Scientific Library through GSL.jl, which is licensed under GPL-3.0, a copyleft license. Users should be aware that GPL terms may apply to the combined work depending on how the software is distributed.

For academic and research use where source code is distributed and users compile the software themselves, which is the typical pattern for scientific computing applications, standard GPL compatibility provisions apply. The MIT-licensed portions and GPL-licensed portions can coexist as separate components that users combine on their own systems.

Users considering commercial distribution or incorporation into proprietary software should consult with legal counsel regarding GPL compliance requirements. The `LICENSE` file contains detailed information about these licensing considerations and the implications for different use cases.

GPU visualization features require NVIDIA CUDA, which is subject to NVIDIA's Software License Agreement. CUDA is freely available for research and academic purposes but has specific terms for commercial use. Users should review NVIDIA's licensing terms if commercial distribution is planned.

---

## Acknowledgments

This software builds upon the excellent work of the Julia community and relies on several key packages that make modern scientific computing in Julia possible. The numerical computations utilize the GNU Scientific Library through the GSL.jl wrapper, which provides battle-tested implementations of special functions and numerical algorithms.

GPU-accelerated visualization is powered by CUDA.jl, which provides Julia bindings for NVIDIA's CUDA toolkit, and GLMakie.jl, which implements high-quality three-dimensional graphics. Additional functionality is provided by Plots.jl for two-dimensional plotting, PrettyTables.jl for formatted console output, TimerOutputs.jl for performance profiling, and ProgressMeter.jl for progress tracking during long calculations.

Complete dependency information including all transitive dependencies is available in the `Project.toml` and `Manifest.toml` files. The Julia community's commitment to open-source scientific computing has made projects like this possible, and I am grateful for their contributions.

---

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| CUDA | 5.9+ | GPU computation |
| Plots | 1.x | 2D plotting |
| GLMakie | 0.13+ | 3D visualization |
| GSL | 1.x | Scientific computations |
| PrettyTables | 3.x | Console output formatting |
| ProgressMeter | 1.x | Progress tracking |

All dependencies are managed in [`Project.toml`](Project.toml) and automatically installed via `setup.jl`.

---

## Citation

If you use this software in your research, please cite it appropriately. A formal citation format will be provided in future releases once the software is published or registered.

---

## Contact

For questions, bug reports, or support inquiries, please contact:

**Email:** ebrajcic@uoguelph.ca

---

**Version**: 0.1.0  
**Julia Version**: 1.11.7  
**Last Updated**: January 2026