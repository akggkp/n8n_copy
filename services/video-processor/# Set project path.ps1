# Set project path
$projectRoot = "C:\Users\akggk\n8n"

# Create all directories
@(
    "services/video-processor/tasks",
    "services/backtesting-service/app/engine",
    "services/ml-service/app/strategy_generator",
    "services/ml-service/app/concept_extractor",
    "services/database",
    "services/frontend/src",
    "services/nginx",
    "services/memory-cleaner",
    "services/analysis",
    "data/videos",
    "data/processed",
    "data/models",
    "data/logs",
    "docs",
    "tests"
) | ForEach-Object {
    $fullPath = Join-Path $projectRoot $_
    New-Item -Path $fullPath -ItemType Directory -Force | Out-Null
    Write-Host "✓ Created: $_" -ForegroundColor Green
}

# Create .env file
@"
POSTGRES_USER=tradingai
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=trading_education
REDIS_PASSWORD=redis_password_here
RABBITMQ_DEFAULT_USER=guest
RABBITMQ_DEFAULT_PASS=guest
N8N_PASSWORD=n8n_admin_password_here
N8N_HOST=localhost
N8N_PORT=5678
DELETE_VIDEO_AFTER_PROCESSING=true
MEMORY_CLEANUP_INTERVAL=300
USE_CASCADE=true
CONFIDENCE_THRESHOLD=0.65
WHISPER_MODEL=base
WHISPER_CASCADE=true
MIN_WIN_RATE_TO_SAVE=55
MIN_PROFIT_FACTOR=1.5
MIN_SHARPE_RATIO=0.5
TIMEZONE=Asia/Kolkata
LOG_LEVEL=INFO
"@ | Out-File -FilePath (Join-Path $projectRoot ".env") -Encoding UTF8

Write-Host "`n✓ Project created at: $projectRoot" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. cd $projectRoot" -ForegroundColor White
Write-Host "2. notepad .env  (update passwords)" -ForegroundColor White
Write-Host "3. Edit files inside services/ folders" -ForegroundColor White
