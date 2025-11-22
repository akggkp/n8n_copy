#!/bin/bash
# Quick diagnostic script to check repository state

echo "=========================================="
echo "Repository Quick Diagnostic"
echo "=========================================="

# Check if critical services exist
echo ""
echo "1. Checking critical services..."
if [ -f "services/video-processor/app/main.py" ]; then
    echo "  ✅ Video Processor EXISTS"
else
    echo "  ❌ Video Processor MISSING"
fi

if [ -f "services/ml-service/app/main.py" ]; then
    echo "  ✅ ML Service EXISTS"
else
    echo "  ❌ ML Service MISSING"
fi

# Check orchestrator tasks completeness
echo ""
echo "2. Checking orchestrator tasks..."
if [ -f "orchestrator/app/tasks.py" ]; then
    pass_count=$(grep -c "^[[:space:]]*pass[[:space:]]*$" orchestrator/app/tasks.py 2>/dev/null || echo "0")
    echo "  Tasks with 'pass' (incomplete): $pass_count"
    if [ "$pass_count" -gt 5 ]; then
        echo "  ⚠️  Many tasks incomplete"
    else
        echo "  ✅ Most tasks implemented"
    fi
else
    echo "  ❌ tasks.py not found"
fi

# Check docker-compose services
echo ""
echo "3. Checking docker-compose..."
if [ -f "docker-compose.yml" ]; then
    service_count=$(grep -c "container_name:" docker-compose.yml 2>/dev/null || echo "0")
    echo "  Services defined: $service_count"
    
    if [ "$service_count" -ge 9 ]; then
        echo "  ✅ All services present"
    else
        echo "  ⚠️  Missing services (expected 9-11)"
    fi
    
    # Check for specific services
    if grep -q "video-processor:" docker-compose.yml; then
        echo "  ✅ video-processor found"
    else
        echo "  ❌ video-processor missing"
    fi
    
    if grep -q "ml-service:" docker-compose.yml; then
        echo "  ✅ ml-service found"
    else
        echo "  ❌ ml-service missing"
    fi
else
    echo "  ❌ docker-compose.yml not found"
fi

# Check database models
echo ""
echo "4. Checking database models..."
if [ -f "orchestrator/app/models.py" ]; then
    model_count=$(grep -c "class.*Base" orchestrator/app/models.py 2>/dev/null || echo "0")
    echo "  Models defined: $model_count"
    
    if [ "$model_count" -ge 6 ]; then
        echo "  ✅ All models present"
    else
        echo "  ⚠️  Missing models (expected 6)"
    fi
else
    echo "  ❌ models.py not found"
fi

# Check critical environment variables
echo ""
echo "5. Checking environment variables..."
if [ -f ".env" ]; then
    if grep -q "VIDEO_PROCESSOR_URL" .env; then
        echo "  ✅ VIDEO_PROCESSOR_URL set"
    else
        echo "  ❌ VIDEO_PROCESSOR_URL missing"
    fi
    
    if grep -q "ML_SERVICE_URL" .env; then
        echo "  ✅ ML_SERVICE_URL set"
    else
        echo "  ❌ ML_SERVICE_URL missing"
    fi
    
    if grep -q "DATABASE_URL" .env; then
        echo "  ✅ DATABASE_URL set"
    else
        echo "  ❌ DATABASE_URL missing"
    fi
else
    echo "  ❌ .env file not found"
fi

echo ""
echo "=========================================="
echo "DIAGNOSTIC COMPLETE"
echo "=========================================="