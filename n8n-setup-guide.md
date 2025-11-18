# n8n Workflow Setup Guide - Trading Education AI

## 📋 Quick Setup (5 Minutes)

### Step 1: Access n8n Dashboard

```
Open your browser: http://localhost:5678
Login: admin / SecureN8nPass123!@#
```

---

## 🔧 Import Workflows

### Workflow 1: Main Video Processing (REQUIRED)

**File:** `n8n-main-workflow.json`

Steps:
1. Click **"+"** button (top left)
2. Select **"Import from File"**
3. Upload `n8n-main-workflow.json`
4. Click **"Import"**
5. Click **"Save"**

**This workflow:**
- ✅ Receives video uploads via webhook
- ✅ Processes videos (extract frames, detect charts, transcribe)
- ✅ Generates trading strategies using Ollama LLM
- ✅ Backtests strategies
- ✅ Saves profitable strategies to database

---

### Workflow 2: Batch Processing (OPTIONAL)

**File:** `n8n-batch-workflow.json`

Steps:
1. Click **"+"** button
2. Select **"Import from File"**
3. Upload `n8n-batch-workflow.json`
4. Click **"Import"**
5. Click **"Save"**

**This workflow:**
- ✅ Runs daily at 2 AM
- ✅ Processes pending videos in batches
- ✅ Calls main workflow for each video
- ✅ Auto-deletes failed videos

---

## 🧪 Test the Workflow

### Option 1: Manual Test (Best for First Time)

1. Open n8n Dashboard
2. Go to **"Main Workflow"**
3. Click **"Test Workflow"**
4. In **"Webhook: Video Upload"** node, input:

```json
{
  "video_id": "test-001",
  "file_path": "/data/videos/sample.mp4",
  "filename": "sample.mp4",
  "file_size_mb": 150
}
```

5. Click **"Send Test Message"**
6. Watch execution in real-time
7. Check each node's output

---

### Option 2: Real API Test (Production)

```bash
curl -X POST http://localhost:5678/webhook/upload-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "prod-001",
    "file_path": "/data/videos/real-video.mp4",
    "filename": "real-video.mp4",
    "file_size_mb": 250
  }'
```

Expected Response:
```json
{
  "status": "success",
  "video_id": "prod-001",
  "message": "Video processing completed",
  "is_profitable": true,
  "timestamp": "2025-11-10T22:25:00.000Z"
}
```

---

## 🔐 Configure Credentials (Optional but Recommended)

### PostgreSQL Connection

1. Go to **Credentials** (bottom left)
2. Click **"+ New"**
3. Select **"Postgres"**
4. Fill in:
   - **Name:** Trading AI Database
   - **Host:** postgres (or localhost)
   - **Port:** 5432
   - **Database:** trading_education
   - **User:** postgres
   - **Password:** SecurePass123
5. Click **"Save"**

---

## 📊 Monitor Workflow Execution

1. Click **"Executions"** tab
2. View all past executions
3. Click any execution to see:
   - ✅ Success/Failed status
   - ⏱️ Total execution time
   - 📝 Each node's input/output
   - 🐛 Error messages

---

## 🚀 Production Deployment

### Enable Workflow Automatically

1. Open workflow
2. Click **"Active"** toggle (top right)
3. Set to **ON**

Now webhook is live and production-ready!

### Webhook URL for External Requests

```
http://localhost:5678/webhook/upload-video
```

Or if exposed externally:

```
https://your-domain.com/webhook/upload-video
```

---

## 🔍 Troubleshooting

### Issue: "Connection refused to video-processor"

**Solution:**
- Check if Python services are running: `docker ps`
- Check service logs: `docker logs trading-video-processor`
- Update URL in HTTP nodes to match your setup

### Issue: "Ollama model not found"

**Solution:**
```bash
# Pull model
docker exec trading-ollama ollama pull llama2

# Verify
docker exec trading-ollama ollama list
```

### Issue: "Database connection failed"

**Solution:**
- Check PostgreSQL is running: `docker logs trading-postgres`
- Verify credentials in nodes
- Test connection: `docker exec trading-postgres psql -U postgres -d trading_education -c "SELECT 1"`

### Issue: Workflow executes but no results saved

**Solution:**
- Check database query in nodes
- Verify PostgreSQL credentials
- Check if tables exist: `docker exec trading-postgres psql -U postgres -d trading_education -c "SELECT * FROM processed_videos LIMIT 1;"`

---

## 📈 Performance Tips

1. **Increase Timeout** (for large videos)
   - In HTTP nodes → Options → Timeout: 600000ms

2. **Parallel Processing**
   - Add multiple video-processor replicas in docker-compose

3. **Caching**
   - Use Redis for caching strategy results
   - Reduce redundant LLM calls

4. **Monitoring**
   - Check execution stats in n8n dashboard
   - Use `docker stats` for resource usage

---

## 📞 Quick Reference

| Component | URL |
|-----------|-----|
| n8n Dashboard | http://localhost:5678 |
| Video Processor API | http://localhost:8000 |
| Backtesting API | http://localhost:8001 |
| Ollama LLM | http://localhost:11434 |
| PostgreSQL | localhost:5432 |
| RabbitMQ | http://localhost:15672 |
| Redis | localhost:6379 |

---

## ✅ Success Checklist

- ✅ n8n running on localhost:5678
- ✅ Workflows imported and saved
- ✅ Video processor running (`docker ps`)
- ✅ PostgreSQL database initialized
- ✅ Ollama with llama2 model
- ✅ Tested webhook (manual or curl)
- ✅ Monitored execution in n8n

**You're ready to process trading videos!** 🎉

---

## 🎓 Next Steps

1. **Upload real trading videos** from your collection
2. **Monitor strategy generation** in Ollama
3. **Review profitable strategies** in database
4. **Adjust parameters** for better accuracy:
   - `confidence_threshold` (currently 0.65)
   - `min_profit_factor` (currently 1.5)
   - `min_win_rate` (currently 55%)

---

**Questions? Check logs:**
```bash
docker logs trading-n8n -f
docker logs trading-video-processor -f
docker logs trading-postgres -f
```