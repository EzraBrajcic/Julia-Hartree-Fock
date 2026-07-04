# License

## MIT License

**Copyright (c) 2026 [Your Name/Institution]**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**

---

## Third-Party Licenses and Notices

This software incorporates or depends upon the following third-party software components, which are subject to their own license terms.

### GNU Scientific Library (GSL)

The GNU Scientific Library is licensed under the GNU General Public License version 3 (GPL-3.0). This software accesses GSL functionality through the GSL.jl wrapper package.

- **Copyright:** Free Software Foundation
- **License:** GPL-3.0
- **Source:** https://www.gnu.org/software/gsl/
- **License Text:** https://www.gnu.org/licenses/gpl-3.0.html

### NVIDIA CUDA Toolkit

NVIDIA CUDA provides GPU acceleration capabilities for visualization features in this software. CUDA is proprietary software owned by NVIDIA Corporation and is subject to the NVIDIA Software License Agreement.

- **Copyright:** NVIDIA Corporation
- **License:** NVIDIA Software License Agreement
- **Source:** https://developer.nvidia.com/cuda-toolkit
- **License Text:** https://docs.nvidia.com/cuda/eula/

### Julia Language Packages

This software depends on several Julia packages that are distributed under the MIT License. These packages include CUDA.jl, GLMakie.jl, Plots.jl, LinearAlgebra.jl, PrettyTables.jl, TimerOutputs.jl, and ProgressMeter.jl. Complete license information for each package is available in their respective repositories at https://github.com/JuliaLang or through https://juliahub.com.

---

## Important GPL Compatibility Notice

This software uses the GNU Scientific Library through the GSL.jl wrapper package. While the original code contained in this repository is licensed under the permissive MIT License, users must understand that GSL itself is licensed under GPL-3.0, which is a copyleft license.

The practical implications of this dual licensing structure depend on how the software is used and distributed. When this software is distributed as source code that users compile and execute themselves, which is the typical pattern for academic and research applications, standard GPL compatibility provisions apply. The MIT-licensed portions and GPL-licensed portions can coexist because they remain separate components that users combine on their own systems.

However, if this software were to be distributed as a compiled binary or incorporated into a larger application that is then distributed, the combined work would likely be subject to GPL-3.0 terms. This distinction is particularly important for users considering commercial distribution or incorporation into proprietary software products.

For academic researchers, educators, and open-source developers using this software in its intended manner by downloading source code and running calculations on their own systems, these GPL considerations do not impose additional restrictions. Users who have questions about GPL compatibility for specific use cases, particularly those involving commercial distribution, should consult with legal counsel familiar with open-source licensing.

---

## Distribution and Compliance

Users who redistribute this software should ensure they comply with the licensing terms of all components. This includes retaining all copyright notices, providing appropriate attribution, and including copies of relevant licenses with any distributed versions of the software. The MIT License permits broad reuse, but proper attribution must always be maintained.

For the GSL component accessed through GSL.jl, users should be aware that GPL-3.0 requires that source code be made available when distributing GPL-licensed software or combined works that include GPL-licensed components. The official GPL-3.0 license text provides complete details on these requirements.

---

**Last Updated:** January 2026