"""Flask web UI for monitoring pipeline"""
import os
import sys
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config
from app.tasks import process_video_pipeline
from app.celery_app import celery_app

app = Flask(__name__)
CORS(app)

# Database
engine = create_engine(Config.DATABASE_URL)
Session = sessionmaker(bind=engine)

@app.route('/')
def index():
    """Dashboard home page"""
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    """Get pipeline statistics"""
    try:
        session = Session()
        
        # Total videos
        total = session.execute(
            text("SELECT COUNT(*) FROM processed_videos")
        ).scalar()
        
        # Completed
        completed = session.execute(
            text("SELECT COUNT(*) FROM processed_videos WHERE status = 'completed'")
        ).scalar()
        
        # Processing
        processing = session.execute(
            text("SELECT COUNT(*) FROM processed_videos WHERE status = 'processing'")
        ).scalar()
        
        # Failed
        failed = session.execute(
            text("SELECT COUNT(*) FROM processed_videos WHERE status = 'failed'")
        ).scalar()
        
        # Profitable strategies
        profitable = session.execute(
            text("SELECT COUNT(*) FROM proven_strategies")
        ).scalar() or 0
        
        session.close()
        
        return jsonify({
            'total_videos': total or 0,
            'completed': completed or 0,
            'processing': processing or 0,
            'failed': failed or 0,
            'profitable_strategies': profitable
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/videos')
def get_videos():
    """Get list of videos with status"""
    try:
        limit = request.args.get('limit', 20, type=int)
        session = Session()
        
        videos = session.execute(
            text("""
                SELECT video_id, filename, status, 
                       processing_time_seconds, processed_at, created_at
                FROM processed_videos
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {'limit': limit}
        ).fetchall()
        
        session.close()
        
        result = []
        for v in videos:
            result.append({
                'video_id': v[0],
                'filename': v[1],
                'status': v[2] or 'pending',
                'processing_time': float(v[3]) if v[3] else None,
                'processed_at': v[4].isoformat() if v[4] else None,
                'created_at': v[5].isoformat() if v[5] else None
            })
        
        return jsonify({'videos': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profitable-strategies')
def get_profitable_strategies():
    """Get list of profitable strategies"""
    try:
        session = Session()
        
        strategies = session.execute(
            text("""
                SELECT video_id, strategy_name, backtest_results, created_at
                FROM proven_strategies
                ORDER BY created_at DESC
                LIMIT 10
            """)
        ).fetchall()
        
        session.close()
        
        result = []
        for s in strategies:
            # Parse backtest results
            import ast
            try:
                results = ast.literal_eval(s[2]) if isinstance(s[2], str) else s[2]
                win_rate = results.get('win_rate', 0)
                profit_factor = results.get('profit_factor', 0)
            except:
                win_rate = 0
                profit_factor = 0
            
            result.append({
                'video_id': s[0],
                'strategy_name': s[1],
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'created_at': s[3].isoformat() if s[3] else None
            })
        
        return jsonify({'strategies': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trigger-processing', methods=['POST'])
def trigger_processing():
    """Manually trigger video processing"""
    try:
        data = request.json
        video_id = data.get('video_id')
        file_path = data.get('file_path')
        filename = data.get('filename')
        
        if not all([video_id, file_path, filename]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Trigger pipeline
        result = process_video_pipeline.delay(video_id, file_path, filename)
        
        return jsonify({
            'status': 'success',
            'task_id': result.id,
            'video_id': video_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/celery-tasks')
def get_celery_tasks():
    """Get active Celery tasks"""
    try:
        # Get active tasks from Celery
        inspect = celery_app.control.inspect()
        active = inspect.active()
        scheduled = inspect.scheduled()
        
        return jsonify({
            'active': active or {},
            'scheduled': scheduled or {}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': Config.SERVICE_NAME,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG
    )