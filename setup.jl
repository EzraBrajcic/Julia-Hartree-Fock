using Pkg

println("\nSetting up Julia Hartree-Fock project...")

# Get the project root directory
project_root = dirname(@__FILE__)

# Function to install all dependencies in the project environment
function install_dependencies()
    println("\nActivating project environment...")
    Pkg.activate(project_root)
    
    println("Updating package registry...")
    Pkg.update()
    
    println("Installing project dependencies...")
    Pkg.instantiate()
    
    println("Precompiling packages (this may take a few minutes)...")
    Pkg.precompile()

    # Force recompilation of CUDA to match system CUDA version
    println("Forcing CUDA recompilation...")
    Pkg.build("CUDA")
end

# Main setup logic
function main()
    println("Project root: $project_root")

    # Install dependencies in project environment
    install_dependencies()

    println("\n" * "="^70)
    println("Setup complete!")
    println("="^70)
    println("\nTo run the project, use:")
    println("  julia --project=. src/main.jl")
    println("\nAll dependencies are installed in the project-specific environment.")
    println()
end

# Run the main setup function
main()