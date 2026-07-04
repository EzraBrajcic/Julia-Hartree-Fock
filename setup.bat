@echo off
:: Start of the script
echo Setting up Julia Hartree-Fock project...

:: Always restart the script with elevated privileges
powershell -Command "if (-not ([Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))) { Start-Process -FilePath 'cmd.exe' -ArgumentList '/c %~dpnx0' -Verb RunAs; exit }"

:: Confirm that the script is running with elevated privileges
echo Running with administrator privileges...

:: Check if Visual C++ Redistributable is installed
echo Checking for Visual C++ Redistributable...
winget list "Microsoft.VCRedist.2015+.x86" >nul 2>&1
if %errorlevel% neq 0 (
    echo Visual C++ Redistributable is not installed. Installing...
    winget install Microsoft.VCRedist.2015+.x86 -e --silent
    
    :: Verify installation
    winget list "Microsoft.VCRedist.2015+.x86" >nul 2>&1
    if %errorlevel% neq 0 (
        echo Warning: Failed to install Visual C++ Redistributable. Continuing anyway...
    ) else (
        echo Visual C++ Redistributable installed successfully!
    )
) else (
    echo Visual C++ Redistributable is already installed.
)

:: Check if Julia is installed
where julia >nul 2>&1
if %errorlevel% neq 0 (
    echo Julia is not installed. Installing Julia...

    :: Define the download URL and output path
    set "julia_url=https://julialang-s3.julialang.org/bin/winnt/x64/1.9/julia-1.9.4-win64.exe"
    set "installer_path=%TEMP%\julia-installer.exe"

    :: Download the Julia installer
    powershell -Command "Invoke-WebRequest -Uri '%julia_url%' -OutFile '%installer_path%'" >nul 2>&1

    :: Run the installer silently
    echo Running the Julia installer...
    "%installer_path%" /S

    :: Verify installation
    where julia >nul 2>&1
    if %errorlevel% neq 0 (
        echo Failed to install Julia. Please install it manually.
        exit /b 1
    )
    echo Julia installed successfully!
) else (
    echo Julia is already installed.
)

:: Run the Julia setup script
echo Running Julia setup script...
julia setup.jl

:: End of script
echo Setup complete!
pause