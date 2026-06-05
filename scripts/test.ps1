pytest -m "not slow and not performance"
$fast_tests = $LASTEXITCODE

pytest -m "slow"
$slow_tests = $LASTEXITCODE

pytest -m "performance"
$performance_tests = $LASTEXITCODE

