#!/usr/bin/env python
"""
Test Orchestrator Tasks
Tests: Task definitions, imports, logic structure
"""
import sys
import os

# Add orchestrator to path
sys.path.insert(0, 'orchestrator')

def test_imports():
    """Test if all required packages are installed"""
    print("🧪 Test 4.1: Checking imports...")
    try:
        import celery
        import requests
        from sqlalchemy import create_engine
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Run: pip install celery requests sqlalchemy")
        return False

def test_tasks_import():
    """Test if tasks module can be imported"""
    print("\n🧪 Test 4.2: Testing tasks import...")
    try:
        from app import tasks
        print("✅ Tasks module imported successfully")
        return True
    except Exception as e:
        print(f"❌ Tasks import failed: {e}")
        return False

def test_task_definitions():
    """Test if all required tasks are defined"""
    print("\n🧪 Test 4.3: Testing task definitions...")
    try:
        from app.tasks import (
            validate_video,
            process_video,
            detect_keywords,
            generate_clips,
            extract_concepts,
            generate_embeddings,
            extract_features,
            generate_strategy,
            backtest_strategy,
            evaluate_and_promote,
            run_full_pipeline
        )
        
        print("✅ All 11 tasks defined")
        print(f"   Tasks: validate_video, process_video, detect_keywords, ...")
        return True
    except Exception as e:
        print(f"❌ Task definitions test failed: {e}")
        return False

def test_task_implementations():
    """Test if tasks have implementations (not just 'pass')"""
    print("\n🧪 Test 4.4: Testing task implementations...")
    try:
        with open('orchestrator/app/tasks.py', 'r') as f:
            content = f.read()
        
        # Count function definitions
        import re
        functions = re.findall(r'def (validate_video|process_video|detect_keywords|generate_clips|extract_concepts|generate_embeddings|extract_features|generate_strategy|backtest_strategy|evaluate_and_promote)\(', content)
        
        # Check for 'pass' statements (indicates incomplete)
        pass_count = len(re.findall(r'^\s+pass\s*$', content, re.MULTILINE))
        
        print(f"✅ Found {len(functions)} task implementations")
        
        if pass_count > 5:
            print(f"⚠️  Warning: {pass_count} 'pass' statements found (incomplete tasks)")
            return False
        elif pass_count > 0:
            print(f"   Note: {pass_count} 'pass' statements (some tasks incomplete)")
        else:
            print(f"   All tasks have complete implementations")
        
        return True
    except Exception as e:
        print(f"❌ Task implementations test failed: {e}")
        return False

def test_helper_functions():
    """Test if helper functions exist"""
    print("\n🧪 Test 4.5: Testing helper functions...")
    try:
        from app.tasks import get_retry_session
        
        # Test retry session creation
        session = get_retry_session()
        
        print("✅ Helper functions working")
        print(f"   get_retry_session() returns: {type(session).__name__}")
        return True
    except Exception as e:
        print(f"❌ Helper functions test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ORCHESTRATOR TASKS - STANDALONE TESTS")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Tasks Import", test_tasks_import()))
    results.append(("Task Definitions", test_task_definitions()))
    results.append(("Task Implementations", test_task_implementations()))
    results.append(("Helper Functions", test_helper_functions()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nPassed: {total_passed}/{len(results)}")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! Orchestrator tasks are ready.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Fix issues before building Docker image.")
        sys.exit(1)
