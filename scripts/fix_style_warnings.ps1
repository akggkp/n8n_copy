#!/usr/bin/env pwsh
# Fix all flake8 style warnings

Write-Host "🔧 Fixing Style Warnings..." -ForegroundColor Cyan

# Function to remove trailing whitespace from file
function Remove-TrailingWhitespace {
    param($FilePath)
    
    $content = Get-Content $FilePath -Raw
    # Remove trailing spaces and tabs from each line
    $fixed = $content -replace '[ \t]+(\r?\n)', '$1'
    # Ensure file ends with newline
    if (-not $fixed.EndsWith("`n")) {
        $fixed += "`n"
    }
    
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($FilePath, $fixed, $utf8)
}

# Fix orchestrator/app/__init__.py (W292 - no newline at end)
Write-Host "Fixing orchestrator/app/__init__.py..." -ForegroundColor Yellow
$file = "orchestrator\app\__init__.py"
if (Test-Path $file) {
    $content = Get-Content $file -Raw
    if (-not $content.EndsWith("`n")) {
        Add-Content $file -Value "" -NoNewline
        "$content`n" | Set-Content $file -NoNewline
    }
    Write-Host "  ✅ Fixed newline at end" -ForegroundColor Green
}

# Fix orchestrator/app/backtest_client.py
Write-Host "Fixing orchestrator/app/backtest_client.py..." -ForegroundColor Yellow
$file = "orchestrator\app\backtest_client.py"
if (Test-Path $file) {
    # Remove unused import and fix whitespace
    $content = Get-Content $file
    
    # Remove 'import json' line (F401 - unused)
    $content = $content | Where-Object { $_ -notmatch '^\s*import json\s*$' }
    
    # Remove trailing whitespace from all lines (W293)
    $content = $content | ForEach-Object { $_.TrimEnd() }
    
    # Join and save
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($file, ($content -join "`n") + "`n", $utf8)
    
    Write-Host "  ✅ Removed unused import" -ForegroundColor Green
    Write-Host "  ✅ Removed trailing whitespace" -ForegroundColor Green
}

# Fix long lines (E501)
Write-Host "Fixing long lines..." -ForegroundColor Yellow
$file = "orchestrator\app\backtest_client.py"
if (Test-Path $file) {
    $content = Get-Content $file
    
    # Fix line 166 and 168 (split long lines)
    $fixed = $content | ForEach-Object {
        $line = $_
        if ($line.Length -gt 127) {
            # Split at appropriate point (after commas, operators)
            if ($line -match '(.*),\s*(.*)') {
                "    " + $matches + ","
                "        " + $matches
            }
            else {
                $line  # Keep as is if can't split cleanly
            }
        }
        else {
            $line
        }
    }
    
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($file, ($fixed -join "`n") + "`n", $utf8)
    Write-Host "  ✅ Fixed long lines" -ForegroundColor Green
}

# Fix orchestrator/app/celery_app.py
Write-Host "Fixing orchestrator/app/celery_app.py..." -ForegroundColor Yellow
$file = "orchestrator\app\celery_app.py"
if (Test-Path $file) {
    $content = Get-Content $file
    
    # E402: Move imports to top
    $imports = @()
    $other = @()
    
    foreach ($line in $content) {
        if ($line -match '^\s*import |^\s*from .* import ') {
            $imports += $line
        }
        else {
            $other += $line
        }
    }
    
    # Rebuild file with imports first
    $newContent = $imports + $other
    
    # Remove trailing whitespace and ensure newline at end
    $newContent = $newContent | ForEach-Object { $_.TrimEnd() }
    
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($file, ($newContent -join "`n") + "`n", $utf8)
    
    Write-Host "  ✅ Moved imports to top" -ForegroundColor Green
    Write-Host "  ✅ Fixed newline at end" -ForegroundColor Green
}

# Fix orchestrator/app/database.py
Write-Host "Fixing orchestrator/app/database.py..." -ForegroundColor Yellow
$file = "orchestrator\app\database.py"
if (Test-Path $file) {
    $content = Get-Content $file
    
    # E302: Add blank line before function (line 35)
    $fixed = @()
    for ($i = 0; $i -lt $content.Length; $i++) {
        $line = $content[$i]
        
        # If this is a function definition and previous line isn't blank
        if ($line -match '^\s*def ' -and $i -gt 0) {
            if ($content[$i-1].Trim() -ne '') {
                $fixed += ''  # Add blank line
            }
        }
        
        # Remove trailing whitespace
        $fixed += $line.TrimEnd()
    }
    
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($file, ($fixed -join "`n") + "`n", $utf8)
    
    Write-Host "  ✅ Added blank lines before functions" -ForegroundColor Green
    Write-Host "  ✅ Removed trailing whitespace" -ForegroundColor Green
}

# Fix orchestrator/app/feature_engineering.py
Write-Host "Fixing orchestrator/app/feature_engineering.py..." -ForegroundColor Yellow
$file = "orchestrator\app\feature_engineering.py"
if (Test-Path $file) {
    Remove-TrailingWhitespace $file
    Write-Host "  ✅ Removed trailing whitespace" -ForegroundColor Green
}

Write-Host "`n✅ All style warnings fixed!" -ForegroundColor Green
Write-Host "Run: flake8 . --count --exclude=.venv" -ForegroundColor Cyan
