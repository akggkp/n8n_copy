# PowerShell - Complete Project Structure Creation Script

## Complete Project Creation Script

Save this as `create-project-structure.ps1` in any directory and run it once:

```powershell
# ============================================================================
# Trading Education AI - Complete Project Structure Creator
# ============================================================================
# This script creates the entire project structure from scratch
# Usage: .\create-project-structure.ps1
# ============================================================================

# Set error action
$ErrorActionPreference = "Continue"

# Colors
$Green = "Green"
$Red = "Red"
$Yellow = "Yellow"
$Cyan = "Cyan"
$White = "White"

# Counters
$CreatedFiles = 0
$CreatedDirs = 0

# Function to create directory
function New-ProjectDirectory {
    param([string]$Path)
    
    if (-not (Test-Path $Path -PathType Container)) {
        New-Item -Path $Path -ItemType Directory -Force | Out-Null
        Write-Host "✓ Created directory: $Path" -ForegroundColor $Green
        $script:CreatedDirs++
    } else {
        Write-Host "• Directory exists: $Path" -ForegroundColor $Gray
    }
}

# Function to create file with content
function New-ProjectFile {
    param(
        [string]$Path,
        [string]$Content = "",
        [string]$Description = ""
    )
    
    $directory = Split-Path $Path
    
    # Ensure directory exists
    if (-not (Test-Path $directory -PathType Container)) {
        New-Item -Path $directory -ItemType Directory -Force | Out-Null
    }
    
    # Create file
    if (-not (Test-Path $Path)) {
        New-Item -Path $Path -ItemType File -Force | Out-Null
        if ($Content) {
            Set-Content -Path $Path -Value $Content -Encoding UTF8
        }
        Write-Host "✓ Created file: $Path" -ForegroundColor $Green
        if ($Description) {
            Write-Host "  └─ $Description" -ForegroundColor $Gray
        }
        $script:CreatedFiles++
    } else {
        Write-Host "• File exists: $Path" -ForegroundColor $Gray
    }
}

# Print header
Write-Host "`n" + ("=" * 80) -ForegroundColor $Cyan
Write-Host "TRADING EDUCATION AI - PROJECT STRUCTURE CREATOR" -ForegroundColor $Cyan
Write-Host ("=" * 80) -ForegroundColor $Cyan
Write-Host "This will create the complete project structure`n" -ForegroundColor $White

# Get project root directory
$projectRoot = Read-Host "Enter project root directory path (default: C:\Users\$env:USERNAME\Desktop\ai-learning\trading-education-ai)"
if ([string]::IsNullOrWhiteSpace($projectRoot)) {
    $projectRoot = "C:\Users\$env:USERNAME\Desktop\ai-learning\trading-education-ai"
}

# Create main directory
Write-Host "`nCreating project in: $projectRoot`n" -ForegroundColor $Cyan

# Create root directory
New-ProjectDirectory $projectRoot
Set-Location $projectRoot

# ============================================================================
# SECTION 1: Root Level Directories
# ============================================================================

Write-Host "`n[SECTION 1] Root Level Directories" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

$rootDirs = @(
    "services",
    "data",
    "data/videos",
    "data/processed",
    "data/models",
    "data/logs",
    "docs",
    ".github"
)

foreach ($dir in $rootDirs) {
    New-ProjectDirectory (Join-Path $projectRoot $dir)
}

# ============================================================================
# SECTION 2: Services - Video Processor
# ============================================================================

Write-Host "`n[SECTION 2] Services - Video Processor" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

$videoProcDirs = @(
    "services/video-processor",
    "services/video-processor/tasks",
    "services/video-processor/app"
)

foreach ($dir in $videoProcDirs) {
    New-ProjectDirectory (Join-Path $projectRoot $dir)
}

# ============================================================================
# SECTION 3: Services - Backtesting
# ============================================================================

Write-Host "`n[SECTION 3] Services - Backtesting" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

$backtestDirs = @(
    "services/backtesting-service",
    "services/backtesting-service/app",
    "services/backtesting-service/app/engine"
)

foreach ($dir in $backtestDirs) {
    New-ProjectDirectory (Join-Path $projectRoot $dir)
}

# ============================================================================
# SECTION 4: Services - ML Service
# ============================================================================

Write-Host "`n[SECTION 4] Services - ML Service" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

$mlDirs = @(
    "services/ml-service",
    "services/ml-service/app",
    "services/ml-service/app/strategy_generator",
    "services/ml-service/app/concept_extractor"
)

foreach ($dir in $mlDirs) {
    New-ProjectDirectory (Join-Path $projectRoot $dir)
}

# ============================================================================
# SECTION 5: Services - Database
# ============================================================================

Write-Host "`n[SECTION 5] Services - Database" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

$dbDirs = @(
    "services/database"
)

foreach ($dir in $dbDirs) {
    New-ProjectDirectory (Join-Path $projectRoot $dir)
}

# ============================================================================
# SECTION 6: Services - Frontend & Nginx
# ============================================================================

Write-Host "`n[SECTION 6] Services - Frontend & Nginx" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

$frontendDirs = @(
    "services/frontend",
    "services/frontend/src",
    "services/frontend/public",
    "services/nginx",
    "services/memory-cleaner"
)

foreach ($dir in $frontendDirs) {
    New-ProjectDirectory (Join-Path $projectRoot $dir)
}

# ============================================================================
# SECTION 7: Analysis & Utils
# ============================================================================

Write-Host "`n[SECTION 7] Analysis & Utilities" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

$analysisDirs = @(
    "services/analysis",
    "tests"
)

foreach ($dir in $analysisDirs) {
    New-ProjectDirectory (Join-Path $projectRoot $dir)
}

# ============================================================================
# SECTION 8: Root Configuration Files
# ============================================================================

Write-Host "`n[SECTION 8] Root Configuration Files" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

# .env file
$envContent = @"
# ============================================================================
# Trading Education AI - Environment Configuration
# ============================================================================

# Database Configuration
POSTGRES_USER=tradingai
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=trading_education

# Redis Configuration
REDIS_PASSWORD=redis_password_here

# RabbitMQ Configuration
RABBITMQ_DEFAULT_USER=guest
RABBITMQ_DEFAULT_PASS=guest

# n8n Configuration
N8N_PASSWORD=n8n_admin_password_here
N8N_HOST=localhost
N8N_PORT=5678

# OpenAlgo Configuration (REMOVED - Using Ollama instead)
# OPENALGO_API_URL=
# OPENALGO_API_KEY=

# Video Processing Configuration
DELETE_VIDEO_AFTER_PROCESSING=true
MEMORY_CLEANUP_INTERVAL=300
USE_CASCADE=true
CONFIDENCE_THRESHOLD=0.65

# Whisper Configuration
WHISPER_MODEL=base
WHISPER_CASCADE=true
WHISPER_CASCADE_MODEL=large

# ML Configuration
BATCH_SIZE=8
LEARNING_RATE=0.001

# Backtesting Configuration
MIN_WIN_RATE_TO_SAVE=55
MIN_PROFIT_FACTOR=1.5
MIN_SHARPE_RATIO=0.5
DELETE_LOSING_STRATEGIES=true

# System Configuration
TIMEZONE=Asia/Kolkata
LOG_LEVEL=INFO
"@

New-ProjectFile -Path (Join-Path $projectRoot ".env") -Content $envContent -Description "Environment variables"

# .gitignore file
$gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Data
data/videos/*
data/processed/*
data/models/*
data/logs/*
!data/.gitkeep

# Docker
.env.local
docker-compose.override.yml

# n8n
.n8n/
n8n_data/

# Testing
.coverage
htmlcov/
.pytest_cache/
.tox/

# OS
Thumbs.db
.DS_Store
*.log

# Keep certain directories
!data/.gitkeep
!tests/.gitkeep
"@

New-ProjectFile -Path (Join-Path $projectRoot ".gitignore") -Content $gitignoreContent -Description "Git ignore rules"

# docker-compose.yml (skeleton)
$dockerComposeContent = @"
version: '3.8'

services:
  # To be populated with full configuration
  # See docker-compose-full.yml for complete version

networks:
  trading-network:
    driver: bridge

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  n8n_data:
    driver: local
"@

New-ProjectFile -Path (Join-Path $projectRoot "docker-compose.yml") -Content $dockerComposeContent -Description "Docker compose basic setup"

# ============================================================================
# SECTION 9: Video Processor Files
# ============================================================================

Write-Host "`n[SECTION 9] Video Processor Service Files" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

# worker.py
$workerContent = @"
# To be populated with video processing logic
# See documentation: cascade-implementation.md
pass
"@

New-ProjectFile -Path (Join-Path $projectRoot "services/video-processor/worker.py") -Content $workerContent -Description "Main video processor worker"

# cascade detector
$cascadeContent = @"
# To be populated with cascade detection logic
# See documentation: cascade-implementation.md
pass
"@

New-ProjectFile -Path (Join-Path $projectRoot "services/video-processor/tasks/chart_detection_cascade.py") -Content $cascadeContent -Description "Cascade chart detection module"

# Dockerfile
$dockerfileContent = @"
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
WORKDIR /app
RUN apt-get update && apt-get install -y python3.11 python3-pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "worker.py"]
"@

New-ProjectFile -Path (Join-Path $projectRoot "services/video-processor/Dockerfile") -Content $dockerfileContent -Description "Docker image for video processor"

# requirements.txt
$requirementsContent = @"
# Core Dependencies
celery==5.3.4
redis==5.0.1
pika==1.3.2

# Video Processing
opencv-python-headless==4.8.1.78
numpy==1.26.2
Pillow==10.1.0

# Audio Processing
openai-whisper==20231117

# Deep Learning
torch==2.1.1
torchvision==0.16.1
ultralytics==8.0.227

# Database
sqlalchemy==2.0.23
psycopg2-binary==2.9.9

# Utils
python-dotenv==1.0.0
psutil==6.0.0
"@

New-ProjectFile -Path (Join-Path $projectRoot "services/video-processor/requirements.txt") -Content $requirementsContent -Description "Python dependencies"

# ============================================================================
# SECTION 10: Backtesting Service Files
# ============================================================================

Write-Host "`n[SECTION 10] Backtesting Service Files" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

# main.py
$backtestMainContent = @"
# To be populated with FastAPI application
# See documentation: cascade-implementation.md
pass
"@

New-ProjectFile -Path (Join-Path $projectRoot "services/backtesting-service/main.py") -Content $backtestMainContent -Description "FastAPI main application"

# backtester.py
$backtesterContent = @"
# To be populated with backtesting engine
# See documentation: optimized-setup.md
pass
"@

New-ProjectFile -Path (Join-Path $projectRoot "services/backtesting-service/app/engine/backtester.py") -Content $backtesterContent -Description "Backtesting engine"

# Dockerfile
$backtestDockerfile = @"
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
"@

New-ProjectFile -Path (Join-Path $projectRoot "services/backtesting-service/Dockerfile") -Content $backtestDockerfile -Description "Docker image for backtesting"

# ============================================================================
# SECTION 11: ML Service Files
# ============================================================================

Write-Host "`n[SECTION 11] ML Training Service Files" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

# ml-service main
$mlMainContent = @"
# To be populated with ML service
# See documentation: n8n-workflows.md
pass
"@

New-ProjectFile -Path (Join-Path $projectRoot "services/ml-service/app/main.py") -Content $mlMainContent -Description "ML service main"

# strategy generator
$strategyContent = @"
# To be populated with strategy generation logic
# See documentation: model-strategy-analysis.md
pass
"@

New-ProjectFile -Path (Join-Path $projectRoot "services/ml-service/app/strategy_generator/model_trainer.py") -Content $strategyContent -Description "Strategy generator"

# ============================================================================
# SECTION 12: Database Files
# ============================================================================

Write-Host "`n[SECTION 12] Database Schema Files" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

# init.sql
$initSqlContent = @"
-- To be populated with database schema
-- See documentation: optimized-setup.md
-- Basic structure created during docker-compose startup
"@

New-ProjectFile -Path (Join-Path $projectRoot "services/database/init.sql") -Content $initSqlContent -Description "Database initialization script"

# cascade migration
$migrationContent = @"
-- To be populated with cascade statistics schema
-- See documentation: cascade-implementation.md
"@

New-ProjectFile -Path (Join-Path $projectRoot "services/database/cascade-migration.sql") -Content $migrationContent -Description "Cascade statistics migration"

# ============================================================================
# SECTION 13: Documentation Files
# ============================================================================

Write-Host "`n[SECTION 13] Documentation Files" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

$docFiles = @(
    "README.md",
    "ARCHITECTURE.md",
    "SETUP_GUIDE.md",
    "API_REFERENCE.md",
    "TROUBLESHOOTING.md"
)

foreach ($doc in $docFiles) {
    $content = "# $($doc -replace '\.md', '')`n`nTo be populated with documentation content`n"
    New-ProjectFile -Path (Join-Path $projectRoot "docs/$doc") -Content $content -Description "Documentation"
}

# ============================================================================
# SECTION 14: n8n Workflow Files
# ============================================================================

Write-Host "`n[SECTION 14] n8n Workflow Files" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

# Main workflow placeholder
$workflowContent = @"
{
  "name": "Trading Education AI - Main Workflow",
  "nodes": [],
  "connections": {},
  "settings": {},
  "staticData": null,
  "tags": [],
  "triggerCount": 1,
  "updatedAt": "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss.000Z')",
  "versionId": "1"
}
"@

New-ProjectFile -Path (Join-Path $projectRoot "trading-education-workflow.json") -Content $workflowContent -Description "Main n8n workflow"

# ============================================================================
# SECTION 15: Utility Scripts
# ============================================================================

Write-Host "`n[SECTION 15] Utility & Verification Scripts" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

# Quick check script
$quickCheckContent = @"
# Trading Education AI - Quick Health Check
# Run regularly to verify all services are running
"@

New-ProjectFile -Path (Join-Path $projectRoot "quick-check.ps1") -Content $quickCheckContent -Description "Quick health check script"

# Progress tracker
$progressContent = @"
# Trading Education AI - Progress Tracker
# Run to see project completion status
"@

New-ProjectFile -Path (Join-Path $projectRoot "progress-tracker.ps1") -Content $progressContent -Description "Progress tracker script"

# ============================================================================
# SECTION 16: Create .gitkeep Files
# ============================================================================

Write-Host "`n[SECTION 16] Creating Directory Markers" -ForegroundColor $Cyan
Write-Host ("─" * 80) -ForegroundColor $Cyan

$gitkeepDirs = @(
    "data",
    "data/videos",
    "data/processed",
    "data/models",
    "data/logs",
    "tests"
)

foreach ($dir in $gitkeepDirs) {
    $gitkeepPath = Join-Path $projectRoot "$dir/.gitkeep"
    if (-not (Test-Path $gitkeepPath)) {
        New-Item -Path $gitkeepPath -ItemType File -Force | Out-Null
        Write-Host "✓ Created marker: $dir/.gitkeep" -ForegroundColor $Green
    }
}

# ============================================================================
# Summary Report
# ============================================================================

Write-Host "`n" + ("=" * 80) -ForegroundColor $Cyan
Write-Host "PROJECT CREATION COMPLETED" -ForegroundColor $Cyan
Write-Host ("=" * 80) -ForegroundColor $Cyan

Write-Host "`nSummary:" -ForegroundColor $Cyan
Write-Host "  Directories created: $CreatedDirs" -ForegroundColor $Green
Write-Host "  Files created: $CreatedFiles" -ForegroundColor $Green
Write-Host "  Project location: $projectRoot" -ForegroundColor $White

Write-Host "`nNext Steps:" -ForegroundColor $Cyan
Write-Host "  1. Navigate to project:" -ForegroundColor $White
Write-Host "     cd $projectRoot`n" -ForegroundColor $Gray

Write-Host "  2. Edit .env file with your settings:" -ForegroundColor $White
Write-Host "     notepad .env`n" -ForegroundColor $Gray

Write-Host "  3. Verify project structure:" -ForegroundColor $White
Write-Host "     dir /s`n" -ForegroundColor $Gray

Write-Host "  4. Read documentation:" -ForegroundColor $White
Write-Host "     notepad docs/README.md`n" -ForegroundColor $Gray

Write-Host "  5. Start populating files:" -ForegroundColor $White
Write-Host "     Follow the 'To be populated' comments in each file`n" -ForegroundColor $Gray

Write-Host "`nFiles Location:" -ForegroundColor $Cyan
Write-Host "  Services: $projectRoot\services\\" -ForegroundColor $Gray
Write-Host "  Data: $projectRoot\data\\" -ForegroundColor $Gray
Write-Host "  Docs: $projectRoot\docs\\" -ForegroundColor $Gray
Write-Host "  Config: $projectRoot\.env" -ForegroundColor $Gray

Write-Host "`n"
```

---

## How to Use This Script

### Step 1: Create the Creator Script

Open PowerShell and run:

```powershell
# Navigate to any directory (like Desktop)
cd C:\Users\$env:USERNAME\Desktop

# Create the creator script
notepad create-project-structure.ps1

# Paste the content above, save and close
```

### Step 2: Allow Script Execution (One-Time)

```powershell
# Set execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Run the Creator Script

```powershell
# Navigate to where you saved the creator script
cd C:\Users\$env:USERNAME\Desktop

# Run it
.\create-project-structure.ps1
```

### Step 4: When Prompted

```
Enter project root directory path (default: C:\Users\akggk\Desktop\ai-learning\trading-education-ai)
# Just press Enter to use default, or type a different path
```

---

## What Gets Created

```
trading-education-ai/
├── services/
│   ├── video-processor/
│   │   ├── tasks/
│   │   │   ├── __init__.py (create manually)
│   │   │   └── chart_detection_cascade.py (placeholder)
│   │   ├── worker.py (placeholder)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── backtesting-service/
│   │   ├── app/
│   │   │   └── engine/
│   │   │       └── backtester.py (placeholder)
│   │   ├── main.py (placeholder)
│   │   └── Dockerfile
│   ├── ml-service/
│   │   ├── app/
│   │   │   ├── strategy_generator/
│   │   │   │   └── model_trainer.py (placeholder)
│   │   │   └── concept_extractor/
│   │   └── main.py (placeholder)
│   ├── database/
│   │   ├── init.sql (placeholder)
│   │   └── cascade-migration.sql (placeholder)
│   ├── frontend/
│   ├── nginx/
│   ├── memory-cleaner/
│   └── analysis/
├── data/
│   ├── videos/ (for video uploads)
│   ├── processed/ (for extracted data)
│   ├── models/ (for ML models)
│   └── logs/ (for application logs)
├── docs/
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── SETUP_GUIDE.md
│   ├── API_REFERENCE.md
│   └── TROUBLESHOOTING.md
├── tests/
├── .env (configuration file)
├── .gitignore
├── docker-compose.yml (skeleton)
├── trading-education-workflow.json (n8n workflow)
├── quick-check.ps1
└── progress-tracker.ps1
```

---

## After Creation: Next Steps

### Fill Files One by One

Each file has a "To be populated" comment. You can now:

```powershell
# 1. Edit .env file
notepad .env
# Add your actual passwords and settings

# 2. Edit docker-compose.yml
notepad docker-compose.yml
# Replace skeleton with full config from docs

# 3. Edit worker.py
notepad services/video-processor/worker.py
# Add the video processing code from documentation

# And so on for each file...
```

### Quick Navigation Commands

```powershell
# Navigate to project
cd C:\Users\akggk\Desktop\ai-learning\trading-education-ai

# View structure
tree /f /a

# Edit any file
notepad services/video-processor/worker.py

# Create Python __init__.py files (needed for packages)
"" | Out-File services/video-processor/__init__.py
"" | Out-File services/video-processor/tasks/__init__.py
"" | Out-File services/backtesting-service/__init__.py
"" | Out-File services/ml-service/__init__.py
```

---

## Summary

✓ This creates **entire project structure** with all directories and placeholder files
✓ All files are ready to edit and populate with code
✓ Configuration files have boilerplate content
✓ Clear comments show where to add code
✓ Can be run once and never again needed

**Total time to create**: ~5 seconds
**Next**: Follow the "To be populated" comments in each file