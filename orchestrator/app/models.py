from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSONB

db = SQLAlchemy()

class ProcessedVideo(db.Model):
    __tablename__ = 'processed_videos'
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(255), unique=True, nullable=False)
    filename = db.Column(db.String(255))
    status = db.Column(db.String(50), default='pending')
    processing_time_seconds = db.Column(db.Numeric(10, 2))
    processed_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    processing_stats = db.Column(JSONB)
    transcription = db.Column(db.Text)
    detected_charts = db.Column(JSONB)

    def to_dict(self):
        return {
            'video_id': self.video_id,
            'filename': self.filename,
            'status': self.status or 'pending',
            'processing_time': float(self.processing_time_seconds) if self.processing_time_seconds else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class ProvenStrategy(db.Model):
    __tablename__ = 'proven_strategies'
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(255), unique=True, nullable=False)
    strategy_name = db.Column(db.String(255), nullable=False)
    strategy_data = db.Column(db.Text)
    backtest_results = db.Column(JSONB)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

    def to_dict(self):
        win_rate = self.backtest_results.get('win_rate', 0) if self.backtest_results else 0
        profit_factor = self.backtest_results.get('profit_factor', 0) if self.backtest_results else 0
        
        return {
            'video_id': self.video_id,
            'strategy_name': self.strategy_name,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class MlStrategy(db.Model):
    __tablename__ = 'ml_strategies'
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(255), unique=True, nullable=False)
    strategy_data = db.Column(db.Text)
    is_profitable = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
