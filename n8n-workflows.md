# Complete n8n Workflow Implementation for Trading Education AI

## Overview

This document provides the complete n8n workflow configuration for your trading education AI platform. The workflow orchestrates video upload, processing, LLM strategy generation, backtesting, and notifications.

---

## Part 1: n8n Workflow JSON Export

### Main Workflow: Trading Education Video Processing

Save this as `trading-education-workflow.json` and import into n8n:

```json
{
  "name": "Trading Education AI - Main Workflow",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "upload-video",
        "responseMode": "responseNode",
        "options": {
          "allowedOrigins": "*"
        }
      },
      "id": "webhook-video-upload",
      "name": "Webhook: Video Upload",
      "type": "n8n-nodes-base.webhook",
      "typeVersion": 1,
      "position": [250, 300],
      "webhookId": "trading-video-upload"
    },
    {
      "parameters": {
        "jsCode": "// Validate video upload\nconst body = $input.first().json.body;\n\nif (!body.video_id || !body.file_path) {\n  throw new Error('Missing video_id or file_path');\n}\n\nif (!body.filename) {\n  throw new Error('Missing filename');\n}\n\n// Check file size (optional)\nconst maxSizeMB = 500;\nif (body.file_size_mb && body.file_size_mb > maxSizeMB) {\n  throw new Error(`File too large: ${body.file_size_mb}MB (max ${maxSizeMB}MB)`);\n}\n\n// Return validated data\nreturn {\n  video_id: body.video_id,\n  file_path: body.file_path,\n  filename: body.filename,\n  file_size_mb: body.file_size_mb || 0,\n  upload_time: new Date().toISOString()\n};"
      },
      "id": "validate-upload",
      "name": "Validate Upload",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [450, 300]
    },
    {
      "parameters": {
        "url": "http://video-processor:8000/process-video",
        "authentication": "none",
        "sendBody": true,
        "specifyBody": "json",
        "jsonBody": "={{ JSON.stringify({\n  video_id: $json.video_id,\n  file_path: $json.file_path,\n  filename: $json.filename,\n  use_cascade: true,\n  confidence_threshold: 0.65\n}) }}",
        "options": {
          "timeout": 600000
        }
      },
      "id": "call-video-processor",
      "name": "Process Video (Python API)",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [650, 300]
    },
    {
      "parameters": {
        "conditions": {
          "options": {
            "caseSensitive": true,
            "leftValue": "",
            "typeValidation": "strict"
          },
          "conditions": [
            {
              "id": "success-check",
              "leftValue": "={{ $json.status }}",
              "rightValue": "success",
              "operator": {
                "type": "string",
                "operation": "equals"
              }
            }
          ],
          "combinator": "and"
        },
        "options": {}
      },
      "id": "check-processing-success",
      "name": "Check Processing Success",
      "type": "n8n-nodes-base.if",
      "typeVersion": 2,
      "position": [850, 300]
    },
    {
      "parameters": {
        "url": "http://ollama:11434/api/generate",
        "authentication": "none",
        "sendBody": true,
        "specifyBody": "json",
        "jsonBody": "={{ JSON.stringify({\n  model: 'llama2',\n  prompt: `You are a trading strategy analyst. Extract trading concepts and generate strategy ideas from this video transcription:\\n\\nTranscription: ${$json.transcription}\\n\\nProvide:\\n1. Key trading concepts mentioned\\n2. Indicators discussed\\n3. Entry/exit rules\\n4. Risk management suggestions\\n\\nFormat as JSON with keys: concepts, indicators, entry_rules, exit_rules, risk_management`,\n  stream: false,\n  format: 'json'\n}) }}",
        "options": {
          "timeout": 30000
        }
      },
      "id": "generate-strategy-ollama",
      "name": "Generate Strategy (Ollama)",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [1050, 250]
    },
    {
      "parameters": {
        "jsCode": "// Parse Ollama response and combine with video data\nconst videoData = $input.first().json;\nconst ollamaResponse = $input.all()[1].json;\n\n// Parse Ollama's JSON response\nlet strategyData;\ntry {\n  strategyData = JSON.parse(ollamaResponse.response);\n} catch (e) {\n  // Fallback if JSON parsing fails\n  strategyData = {\n    concepts: [],\n    indicators: [],\n    entry_rules: ollamaResponse.response,\n    exit_rules: '',\n    risk_management: ''\n  };\n}\n\n// Combine video data with strategy\nreturn {\n  video_id: videoData.video_id,\n  filename: videoData.filename,\n  transcription: videoData.transcription,\n  detected_charts: videoData.detected_charts,\n  processing_stats: videoData.processing_stats,\n  strategy: {\n    concepts: strategyData.concepts || [],\n    indicators: strategyData.indicators || [],\n    entry_rules: strategyData.entry_rules || '',\n    exit_rules: strategyData.exit_rules || '',\n    risk_management: strategyData.risk_management || '',\n    generated_at: new Date().toISOString()\n  }\n};"
      },
      "id": "parse-strategy",
      "name": "Parse Strategy",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [1250, 250]
    },
    {
      "parameters": {
        "url": "http://backtesting-service:8001/backtest",
        "authentication": "none",
        "sendBody": true,
        "specifyBody": "json",
        "jsonBody": "={{ JSON.stringify({\n  video_id: $json.video_id,\n  strategy: $json.strategy,\n  symbol: 'NIFTY',\n  timeframe: '15m',\n  start_date: '2024-01-01',\n  end_date: '2024-12-31'\n}) }}",
        "options": {
          "timeout": 300000
        }
      },
      "id": "run-backtest",
      "name": "Run Backtest (Python API)",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [1450, 250]
    },
    {
      "parameters": {
        "conditions": {
          "options": {
            "caseSensitive": true,
            "leftValue": "",
            "typeValidation": "strict"
          },
          "conditions": [
            {
              "id": "profitable-check",
              "leftValue": "={{ $json.backtest_results.is_profitable }}",
              "rightValue": "true",
              "operator": {
                "type": "boolean",
                "operation": "true"
              }
            }
          ],
          "combinator": "and"
        }
      },
      "id": "check-profitability",
      "name": "Check if Profitable",
      "type": "n8n-nodes-base.if",
      "typeVersion": 2,
      "position": [1650, 250]
    },
    {
      "parameters": {
        "operation": "executeQuery",
        "query": "INSERT INTO proven_strategies (video_id, strategy_name, strategy_data, backtest_results, created_at)\nVALUES (\n  '{{ $json.video_id }}',\n  '{{ $json.strategy.indicators.join('_') }}_Strategy',\n  '{{ JSON.stringify($json.strategy) }}',\n  '{{ JSON.stringify($json.backtest_results) }}',\n  NOW()\n);",
        "options": {}
      },
      "id": "save-profitable-strategy",
      "name": "Save Profitable Strategy",
      "type": "n8n-nodes-base.postgres",
      "typeVersion": 2.4,
      "position": [1850, 150],
      "credentials": {
        "postgres": {
          "id": "postgres-credentials",
          "name": "Trading AI Database"
        }
      }
    },
    {
      "parameters": {
        "operation": "executeQuery",
        "query": "DELETE FROM ml_strategies WHERE video_id = '{{ $json.video_id }}' AND is_profitable = false;",
        "options": {}
      },
      "id": "delete-unprofitable",
      "name": "Delete Unprofitable Strategy",
      "type": "n8n-nodes-base.postgres",
      "typeVersion": 2.4,
      "position": [1850, 350],
      "credentials": {
        "postgres": {
          "id": "postgres-credentials",
          "name": "Trading AI Database"
        }
      }
    },
    {
      "parameters": {
        "jsCode": "// Create success notification\nconst results = $input.first().json;\n\nlet message;\nif (results.backtest_results.is_profitable) {\n  message = `✅ Profitable Strategy Found!\\n\\nVideo: ${results.filename}\\nStrategy: ${results.strategy.indicators.join(', ')}\\nWin Rate: ${results.backtest_results.win_rate}%\\nProfit Factor: ${results.backtest_results.profit_factor}\\nSharpe Ratio: ${results.backtest_results.sharpe_ratio}\\nTotal P&L: $${results.backtest_results.total_pnl}`;\n} else {\n  message = `❌ Strategy Not Profitable\\n\\nVideo: ${results.filename}\\nWin Rate: ${results.backtest_results.win_rate}%\\nStrategy deleted.`;\n}\n\nreturn {\n  message: message,\n  video_id: results.video_id,\n  is_profitable: results.backtest_results.is_profitable,\n  timestamp: new Date().toISOString()\n};"
      },
      "id": "format-notification",
      "name": "Format Notification",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [2050, 250]
    },
    {
      "parameters": {
        "resource": "message",
        "channel": "#trading-strategies",
        "text": "={{ $json.message }}",
        "attachments": [],
        "otherOptions": {}
      },
      "id": "send-slack-notification",
      "name": "Send Slack Notification",
      "type": "n8n-nodes-base.slack",
      "typeVersion": 2.1,
      "position": [2250, 250],
      "credentials": {
        "slackApi": {
          "id": "slack-credentials",
          "name": "Slack API"
        }
      }
    },
    {
      "parameters": {
        "respondWith": "json",
        "responseBody": "={{ JSON.stringify({\n  status: 'success',\n  video_id: $json.video_id,\n  message: 'Video processing completed',\n  is_profitable: $json.is_profitable,\n  processing_time: $json.processing_time\n}) }}"
      },
      "id": "webhook-response",
      "name": "Webhook Response",
      "type": "n8n-nodes-base.respondToWebhook",
      "typeVersion": 1,
      "position": [2450, 250]
    },
    {
      "parameters": {
        "jsCode": "// Handle error\nconst error = $input.first().json;\n\nreturn {\n  status: 'error',\n  video_id: error.video_id || 'unknown',\n  error_message: error.message || 'Processing failed',\n  timestamp: new Date().toISOString()\n};"
      },
      "id": "handle-error",
      "name": "Handle Error",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [850, 450]
    },
    {
      "parameters": {
        "respondWith": "json",
        "responseBody": "={{ JSON.stringify({\n  status: 'error',\n  video_id: $json.video_id,\n  error: $json.error_message\n}) }}"
      },
      "id": "error-response",
      "name": "Error Response",
      "type": "n8n-nodes-base.respondToWebhook",
      "typeVersion": 1,
      "position": [1050, 450]
    }
  ],
  "connections": {
    "Webhook: Video Upload": {
      "main": [
        [
          {
            "node": "Validate Upload",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Validate Upload": {
      "main": [
        [
          {
            "node": "Process Video (Python API)",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Process Video (Python API)": {
      "main": [
        [
          {
            "node": "Check Processing Success",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Check Processing Success": {
      "main": [
        [
          {
            "node": "Generate Strategy (Ollama)",
            "type": "main",
            "index": 0
          }
        ],
        [
          {
            "node": "Handle Error",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Generate Strategy (Ollama)": {
      "main": [
        [
          {
            "node": "Parse Strategy",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Parse Strategy": {
      "main": [
        [
          {
            "node": "Run Backtest (Python API)",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Run Backtest (Python API)": {
      "main": [
        [
          {
            "node": "Check if Profitable",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Check if Profitable": {
      "main": [
        [
          {
            "node": "Save Profitable Strategy",
            "type": "main",
            "index": 0
          }
        ],
        [
          {
            "node": "Delete Unprofitable Strategy",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Save Profitable Strategy": {
      "main": [
        [
          {
            "node": "Format Notification",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Delete Unprofitable Strategy": {
      "main": [
        [
          {
            "node": "Format Notification",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Format Notification": {
      "main": [
        [
          {
            "node": "Send Slack Notification",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Send Slack Notification": {
      "main": [
        [
          {
            "node": "Webhook Response",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Handle Error": {
      "main": [
        [
          {
            "node": "Error Response",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  },
  "settings": {
    "executionOrder": "v1"
  },
  "staticData": null,
  "tags": [],
  "triggerCount": 1,
  "updatedAt": "2025-11-04T00:00:00.000Z",
  "versionId": "1"
}
```

---

## Part 2: Scheduled Batch Processing Workflow

For processing videos in batches (e.g., every night at 2 AM):

```json
{
  "name": "Trading Education AI - Scheduled Batch Processing",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [
            {
              "field": "cronExpression",
              "expression": "0 2 * * *"
            }
          ]
        }
      },
      "id": "schedule-trigger",
      "name": "Schedule: Daily 2 AM",
      "type": "n8n-nodes-base.scheduleTrigger",
      "typeVersion": 1.1,
      "position": [250, 300]
    },
    {
      "parameters": {
        "operation": "executeQuery",
        "query": "SELECT video_id, file_path, filename FROM videos WHERE status = 'uploaded' AND processed_at IS NULL ORDER BY upload_time ASC LIMIT 10;",
        "options": {}
      },
      "id": "fetch-pending-videos",
      "name": "Fetch Pending Videos",
      "type": "n8n-nodes-base.postgres",
      "typeVersion": 2.4,
      "position": [450, 300],
      "credentials": {
        "postgres": {
          "id": "postgres-credentials",
          "name": "Trading AI Database"
        }
      }
    },
    {
      "parameters": {
        "batchSize": 1,
        "options": {}
      },
      "id": "split-into-batches",
      "name": "Split into Batches",
      "type": "n8n-nodes-base.splitInBatches",
      "typeVersion": 3,
      "position": [650, 300]
    },
    {
      "parameters": {
        "url": "http://localhost:5678/webhook/upload-video",
        "authentication": "none",
        "sendBody": true,
        "specifyBody": "json",
        "jsonBody": "={{ JSON.stringify({\n  video_id: $json.video_id,\n  file_path: $json.file_path,\n  filename: $json.filename\n}) }}",
        "options": {}
      },
      "id": "trigger-main-workflow",
      "name": "Trigger Main Workflow",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [850, 300]
    },
    {
      "parameters": {
        "amount": 30,
        "unit": "seconds"
      },
      "id": "wait-between-videos",
      "name": "Wait Between Videos",
      "type": "n8n-nodes-base.wait",
      "typeVersion": 1.1,
      "position": [1050, 300]
    },
    {
      "parameters": {
        "jsCode": "// Summary of batch processing\nconst totalVideos = $input.all().length;\nconst successCount = $input.all().filter(x => x.json.status === 'success').length;\n\nreturn {\n  total_videos: totalVideos,\n  successful: successCount,\n  failed: totalVideos - successCount,\n  timestamp: new Date().toISOString()\n};"
      },
      "id": "batch-summary",
      "name": "Batch Summary",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [1250, 300]
    }
  ],
  "connections": {
    "Schedule: Daily 2 AM": {
      "main": [
        [
          {
            "node": "Fetch Pending Videos",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Fetch Pending Videos": {
      "main": [
        [
          {
            "node": "Split into Batches",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Split into Batches": {
      "main": [
        [
          {
            "node": "Trigger Main Workflow",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Trigger Main Workflow": {
      "main": [
        [
          {
            "node": "Wait Between Videos",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Wait Between Videos": {
      "main": [
        [
          {
            "node": "Split into Batches",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  },
  "settings": {},
  "staticData": null,
  "tags": []
}
```

---

## Part 3: Setup Instructions

### Step 1: Install n8n

```bash
# Option 1: Docker (Recommended)
docker run -d \
  --name n8n \
  -p 5678:5678 \
  -e N8N_BASIC_AUTH_ACTIVE=true \
  -e N8N_BASIC_AUTH_USER=admin \
  -e N8N_BASIC_AUTH_PASSWORD=your_password \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n

# Option 2: npm
npm install -g n8n
n8n start
```

### Step 2: Access n8n Dashboard

```
Open browser: http://localhost:5678
Login with credentials
```

### Step 3: Configure Credentials

#### PostgreSQL Credentials
```
Name: Trading AI Database
Type: Postgres
Host: postgres (or localhost if not using Docker network)
Port: 5432
Database: trading_education
User: tradingai
Password: your_password
```

#### Slack Credentials (Optional)
```
Name: Slack API
Type: Slack
OAuth2 Access Token: xoxb-your-token
```

### Step 4: Import Workflows

```
1. Click "+" button in n8n
2. Click "Import from File"
3. Upload "trading-education-workflow.json"
4. Click "Import"
5. Repeat for batch processing workflow
```

### Step 5: Update Service URLs

In each HTTP Request node, update URLs to match your setup:

```javascript
// If using Docker Compose network
http://video-processor:8000/process-video
http://backtesting-service:8001/backtest
http://ollama:11434/api/generate

// If running locally without Docker network
http://localhost:8000/process-video
http://localhost:8001/backtest
http://localhost:11434/api/generate
```

### Step 6: Test Webhook

```bash
# Get webhook URL from n8n
# It will be: http://localhost:5678/webhook/upload-video

# Test with curl
curl -X POST http://localhost:5678/webhook/upload-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "test-123",
    "file_path": "/data/videos/test-video.mp4",
    "filename": "test-video.mp4",
    "file_size_mb": 100
  }'
```

---

## Part 4: Complete docker-compose.yml with n8n

```yaml
version: '3.8'

services:
  # ==================== n8n Orchestrator ====================
  
  n8n:
    image: n8nio/n8n:latest
    container_name: trading-n8n
    restart: unless-stopped
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - N8N_HOST=localhost
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      - WEBHOOK_URL=http://localhost:5678/
      - GENERIC_TIMEZONE=Asia/Kolkata
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_PORT=5432
      - DB_POSTGRESDB_DATABASE=n8n
      - DB_POSTGRESDB_USER=${POSTGRES_USER}
      - DB_POSTGRESDB_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - n8n_data:/home/node/.n8n
      - ./data/videos:/data/videos:ro
    depends_on:
      - postgres
    networks:
      - trading-network
    mem_limit: 512m

  # ==================== Database ====================
  
  postgres:
    image: postgres:16-alpine
    container_name: trading-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: trading_education
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./services/database/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - trading-network
    mem_limit: 1.5g

  # ==================== Python Services ====================
  
  video-processor:
    build:
      context: ./services/video-processor
      dockerfile: Dockerfile
    container_name: trading-video-processor
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/trading_education
      - USE_CASCADE=true
      - CONFIDENCE_THRESHOLD=0.65
    volumes:
      - ./data/videos:/data/videos
      - ./data/processed:/data/processed
      - ./data/models:/app/models
    depends_on:
      - postgres
    networks:
      - trading-network
    mem_limit: 2.5g
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  backtesting-service:
    build:
      context: ./services/backtesting-service
      dockerfile: Dockerfile
    container_name: trading-backtesting
    restart: unless-stopped
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/trading_education
      - MIN_WIN_RATE_TO_SAVE=55
      - MIN_PROFIT_FACTOR=1.5
    depends_on:
      - postgres
    networks:
      - trading-network
    mem_limit: 1g

  # ==================== Ollama LLM ====================
  
  ollama:
    image: ollama/ollama:latest
    container_name: trading-ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - OLLAMA_MODELS=/root/.ollama/models
    volumes:
      - ollama_data:/root/.ollama
    networks:
      - trading-network
    mem_limit: 4.5g
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # ==================== Redis & RabbitMQ (Optional) ====================
  
  redis:
    image: redis:7-alpine
    container_name: trading-redis
    restart: unless-stopped
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    networks:
      - trading-network
    mem_limit: 256m

networks:
  trading-network:
    driver: bridge

volumes:
  postgres_data:
    driver: local
  n8n_data:
    driver: local
  ollama_data:
    driver: local
```

---

## Part 5: Python FastAPI Endpoints

### services/video-processor/main.py

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tasks.chart_detection_cascade import CascadeChartDetector
import logging

app = FastAPI(title="Video Processor API")
logger = logging.getLogger(__name__)

class VideoProcessRequest(BaseModel):
    video_id: str
    file_path: str
    filename: str
    use_cascade: bool = True
    confidence_threshold: float = 0.65

class VideoProcessResponse(BaseModel):
    status: str
    video_id: str
    transcription: str
    detected_charts: list
    processing_stats: dict
    message: str

@app.post("/process-video", response_model=VideoProcessResponse)
async def process_video(request: VideoProcessRequest):
    """
    Process video: extract frames, detect charts, transcribe audio
    """
    try:
        logger.info(f"Processing video: {request.video_id}")
        
        # Import processor
        from worker import VideoProcessorCascade
        
        processor = VideoProcessorCascade()
        success = processor.process_video(request.video_id, request.file_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Processing failed")
        
        # Fetch results from database
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        import os
        
        DATABASE_URL = os.getenv("DATABASE_URL")
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Query processed video
        result = session.execute(
            "SELECT transcription, detected_charts, processing_stats FROM processed_videos WHERE video_id = :vid",
            {"vid": request.video_id}
        ).fetchone()
        
        session.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Processed video not found")
        
        return VideoProcessResponse(
            status="success",
            video_id=request.video_id,
            transcription=result[0],
            detected_charts=result[1],
            processing_stats=result[2],
            message="Video processed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "video-processor"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### services/backtesting-service/main.py

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

app = FastAPI(title="Backtesting API")
logger = logging.getLogger(__name__)

class BacktestRequest(BaseModel):
    video_id: str
    strategy: Dict[str, Any]
    symbol: str = "NIFTY"
    timeframe: str = "15m"
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"

class BacktestResponse(BaseModel):
    status: str
    video_id: str
    backtest_results: Dict[str, Any]
    message: str

@app.post("/backtest", response_model=BacktestResponse)
async def backtest_strategy(request: BacktestRequest):
    """
    Backtest strategy and return results
    """
    try:
        logger.info(f"Backtesting strategy for video: {request.video_id}")
        
        # Import backtester
        from engine.backtester import BacktestEngine
        
        engine = BacktestEngine()
        
        # Run backtest
        is_profitable = engine.backtest_strategy(
            strategy_id=request.video_id,
            strategy_data=request.strategy,
            historical_data=None  # Fetch from database or API
        )
        
        # Get results from database
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        import os
        
        DATABASE_URL = os.getenv("DATABASE_URL")
        db_engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=db_engine)
        session = Session()
        
        # Query backtest results
        result = session.execute(
            """SELECT win_rate, profit_factor, sharpe_ratio, total_pnl, 
                      total_trades, winning_trades, losing_trades 
               FROM backtest_results 
               WHERE strategy_id = :sid 
               ORDER BY tested_at DESC LIMIT 1""",
            {"sid": request.video_id}
        ).fetchone()
        
        session.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Backtest results not found")
        
        return BacktestResponse(
            status="success",
            video_id=request.video_id,
            backtest_results={
                "is_profitable": is_profitable,
                "win_rate": float(result[0]),
                "profit_factor": float(result[1]),
                "sharpe_ratio": float(result[2]),
                "total_pnl": float(result[3]),
                "total_trades": result[4],
                "winning_trades": result[5],
                "losing_trades": result[6]
            },
            message="Backtest completed"
        )
        
    except Exception as e:
        logger.error(f"Error backtesting: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "backtesting"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

---

## Part 6: Testing the Complete Setup

### Step 1: Start All Services

```bash
# Start Docker Compose
docker-compose up -d

# Check all services are running
docker-compose ps

# Check logs
docker logs trading-n8n -f
```

### Step 2: Load Ollama Model

```bash
# Enter Ollama container
docker exec -it trading-ollama bash

# Pull LLaMA 2 model
ollama pull llama2

# Test model
ollama run llama2 "What is a trading strategy?"

# Exit container
exit
```

### Step 3: Test Python APIs

```bash
# Test video processor health
curl http://localhost:8000/health

# Test backtesting health
curl http://localhost:8001/health

# Test Ollama
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "What is RSI indicator?",
  "stream": false
}'
```

### Step 4: Upload Test Video via n8n

```bash
# Test webhook
curl -X POST http://localhost:5678/webhook/upload-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "test-video-001",
    "file_path": "/data/videos/sample.mp4",
    "filename": "sample.mp4",
    "file_size_mb": 150
  }'
```

### Step 5: Monitor Execution in n8n

```
1. Open n8n: http://localhost:5678
2. Click "Executions" tab
3. Watch real-time execution
4. Check each node's output
5. Verify Ollama response
6. Verify backtest results
```

---

## Part 7: Expected Workflow Timeline

```
Video Upload → Webhook (0.1s)
    ↓
Validate Upload → Code (0.2s)
    ↓
Process Video → Python API (120-180s)
    ├─ Frame extraction (30s)
    ├─ Cascade detection (60s)
    └─ Transcription (30s)
    ↓
Generate Strategy → Ollama (2-3s)
    ↓
Parse Strategy → Code (0.1s)
    ↓
Run Backtest → Python API (30-60s)
    ↓
Check Profitability → If (0.1s)
    ↓
Save/Delete → PostgreSQL (0.5s)
    ↓
Notification → Slack (1s)
    ↓
Response → Webhook (0.1s)

Total Time: ~3-5 minutes per video
```

---

## Summary

You now have:
1. ✓ Complete n8n workflow JSON (copy-paste ready)
2. ✓ Scheduled batch processing workflow
3. ✓ Docker Compose with n8n + Python + Ollama
4. ✓ Python FastAPI endpoints for both services
5. ✓ Testing procedures
6. ✓ Timeline expectations

**Next steps**: Import workflows into n8n, test with sample video, scale to full library!