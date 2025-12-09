# check.ps1
Write-Host "Checking code format..." -ForegroundColor Cyan
black --check .
if ($LASTEXITCODE -ne 0) { 
    Write-Host "❌ Format failed! Run 'black .' to fix" -ForegroundColor Red
    exit 1 
}

Write-Host "Running linter..." -ForegroundColor Cyan
ruff check .
if ($LASTEXITCODE -ne 0) { 
    Write-Host "❌ Linting failed!" -ForegroundColor Red
    exit 1 
}

Write-Host "Running tests..." -ForegroundColor Cyan
pytest -q
if ($LASTEXITCODE -ne 0) { 
    Write-Host "❌ Tests failed!" -ForegroundColor Red
    exit 1 
}

Write-Host "✅ All checks passed!" -ForegroundColor Green