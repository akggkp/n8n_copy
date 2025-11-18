"""Flask web UI for monitoring pipeline"""
import os
import sys
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config
from app.tasks import process_video_pipeline
from app.celery_app import celery_app
from app.models import db, ProcessedVideo, ProvenStrategy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = Config.DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = Config.VIDEO_WATCH_DIR


db.init_app(app)
CORS(app)

@app.route('/')
def index():
    """Dashboard home page"""
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    """Get pipeline statistics"""
    try:
        total = db.session.query(ProcessedVideo).count()
        completed = db.session.query(ProcessedVideo).filter_by(status='completed').count()
        processing = db.session.query(ProcessedVideo).filter_by(status='processing').count()
        failed = db.session.query(ProcessedVideo).filter_by(status='failed').count()
        profitable = db.session.query(ProvenStrategy).count()
        
        return jsonify({
            'total_videos': total,
            'completed': completed,
            'processing': processing,
            'failed': failed,
            'profitable_strategies': profitable
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/videos')
def get_videos():
    """Get list of videos with status"""
    try:
        limit = request.args.get('limit', 20, type=int)
        videos = db.session.query(ProcessedVideo).order_by(ProcessedVideo.created_at.desc()).limit(limit).all()
        
        return jsonify({'videos': [v.to_dict() for v in videos]})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profitable-strategies')
def get_profitable_strategies():
    """Get list of profitable strategies"""
    try:
        strategies = db.session.query(ProvenStrategy).order_by(ProvenStrategy.created_at.desc()).limit(10).all()
        
        return jsonify({'strategies': [s.to_dict() for s in strategies]})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        video_id = f'upload-{int(datetime.now().timestamp())}'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Trigger pipeline
        result = process_video_pipeline.delay(video_id, file_path, filename)

        return jsonify({
            'status': 'success',
            'task_id': result.id,
            'video_id': video_id,
            'filename': filename,
            'message': 'File uploaded and processing started.'
        }), 200

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
    with app.app_context():
        db.create_all()
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG
    )