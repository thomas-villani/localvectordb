# Check if the "src" folder exists in the current directory
if (Test-Path "./src" -PathType Container) {
    Write-Host "Setting PYTHONPATH to './src'"
    $env:PYTHONPATH = "./src"
} else {
    Write-Host "Directory './src' not found. PYTHONPATH not set."
}

# Define a list of common virtual environment directory names
$venvPaths = @("./.venv", "./venv", "./env", "./virtualenv")

if -not [string]::IsNullOrEmpty($env:$VIRTUAL_ENV) {
    Write-Host "venv already activated."
}

# Initialize a flag to indicate if a virtual environment was activated
$venvActivated = $false

# Iterate over the list of virtual environment paths
foreach ($venvPath in $venvPaths) {
    if (Test-Path $venvPath -PathType Container) {
        # Activate the virtual environment by running the activate script
        Write-Host "Activating the virtual environment in '$venvPath'..."
        . "$venvPath/scripts/activate.ps1"
        $venvActivated = $true
        break
    }
}

if (-not $venvActivated) {
    Write-Host "No virtual environment found."
}
