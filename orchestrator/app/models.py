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
            'processing_time': float(
                self.processing_time_seconds) if self.processing_time_seconds else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None}


class ProvenStrategy(db.Model):
    __tablename__ = "proven_strategies"

    id = db.Column(db.Integer, primary_key=True)
    media_item_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "processed_videos.id",
            ondelete="CASCADE"))  # Link to ProcessedVideo.id
    strategy_id = db.Column(db.Integer)  # From backtesting service

    # Performance metrics
    sharpe_ratio = db.Column(db.Float)
    win_rate = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    total_return = db.Column(db.Float)
    num_trades = db.Column(db.Integer)
    avg_trade_duration = db.Column(db.Float)  # hours

    # Status
    # promoted, active, retired
    status = db.Column(db.String(50), default='promoted')
    promoted_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    # Relationships
    media_item = db.relationship("ProcessedVideo", backref="proven_strategies")


class MlStrategy(db.Model):
    __tablename__ = 'ml_strategies'
    id = db.Column(db.Integer, primary_key=True)
    media_item_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "media_items.id",
            ondelete="CASCADE"),
        unique=True,
        nullable=False)  # Changed from video_id
    strategy_data = db.Column(db.Text)
    is_profitable = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# ==================== Unified Data Models ====================
# These models are consistent with services/api-service/app/models.py
# and orchestrator/app/tasks.py needs them for feature extraction.


class MediaItem(db.Model):
    __tablename__ = "media_items"

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(255), unique=True, nullable=False)
    source_url = db.Column(db.Text)
    filename = db.Column(db.String(255), nullable=False)
    duration_seconds = db.Column(db.Float)
    file_size_bytes = db.Column(db.Integer)
    status = db.Column(db.String(50), default='pending')
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(
        db.DateTime,
        default=db.func.current_timestamp(),
        onupdate=db.func.current_timestamp())

    # Relationships
    transcripts = db.relationship(
        "Transcript",
        back_populates="media_item",
        cascade="all, delete-orphan")
    keyword_hits = db.relationship(
        "KeywordHit",
        back_populates="media_item",
        cascade="all, delete-orphan")
    frames = db.relationship(
        "Frame",
        back_populates="media_item",
        cascade="all, delete-orphan")
    clips = db.relationship(
        "Clip",
        back_populates="media_item",
        cascade="all, delete-orphan")
    embeddings = db.relationship(
        "Embedding",
        back_populates="media_item",
        cascade="all, delete-orphan")


class Transcript(db.Model):
    __tablename__ = "transcripts"

    id = db.Column(db.Integer, primary_key=True)
    media_item_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "media_items.id",
            ondelete="CASCADE"))
    segment_index = db.Column(db.Integer, nullable=False)
    start_time = db.Column(db.Float, nullable=False)
    end_time = db.Column(db.Float, nullable=False)
    text = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(10), default="en")
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    media_item = db.relationship("MediaItem", back_populates="transcripts")


class KeywordHit(db.Model):
    __tablename__ = "keyword_hits"

    id = db.Column(db.Integer, primary_key=True)
    media_item_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "media_items.id",
            ondelete="CASCADE"))
    keyword = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50))
    start_time = db.Column(db.Float, nullable=False)
    end_time = db.Column(db.Float)
    confidence = db.Column(db.Float, default=1.0)
    context_text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    media_item = db.relationship("MediaItem", back_populates="keyword_hits")
    clips = db.relationship("Clip", back_populates="keyword_hit")


class Clip(db.Model):
    __tablename__ = "clips"

    id = db.Column(db.Integer, primary_key=True)
    media_item_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "media_items.id",
            ondelete="CASCADE"))
    keyword_hit_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "keyword_hits.id",
            ondelete="SET NULL"))
    keyword = db.Column(db.String(100))
    start_time = db.Column(db.Float, nullable=False)
    end_time = db.Column(db.Float, nullable=False)
    duration_seconds = db.Column(db.Float)
    file_path = db.Column(db.Text, nullable=False)
    file_size_bytes = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    media_item = db.relationship("MediaItem", back_populates="clips")
    keyword_hit = db.relationship("KeywordHit", back_populates="clips")


class Frame(db.Model):
    __tablename__ = "frames"

    id = db.Column(db.Integer, primary_key=True)
    media_item_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "media_items.id",
            ondelete="CASCADE"))
    timestamp = db.Column(db.Float, nullable=False)
    file_path = db.Column(db.Text, nullable=False)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    contains_chart = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    media_item = db.relationship("MediaItem", back_populates="frames")


class Embedding(db.Model):
    __tablename__ = "embeddings"

    id = db.Column(db.Integer, primary_key=True)
    media_item_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "media_items.id",
            ondelete="CASCADE"))
    embedding_type = db.Column(db.String(50))
    reference_id = db.Column(db.Integer)
    embedding_model = db.Column(db.String(100))
    # Using JSONB for ARRAY(Float) in Flask-SQLAlchemy
    embedding_vector = db.Column(JSONB)
    vector_dimension = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    media_item = db.relationship("MediaItem", back_populates="embeddings")


class StrategyFeature(db.Model):  # Renamed from StrategiesFeature
    __tablename__ = 'strategies_features'
    id = db.Column(db.Integer, primary_key=True)
    media_item_id = db.Column(
        db.Integer,
        db.ForeignKey(
            'media_items.id',
            ondelete='CASCADE'),
        nullable=False)
    strategy_id = db.Column(db.String(255))
    feature_vector = db.Column(JSONB, nullable=False)
    feature_names = db.Column(db.ARRAY(db.Text))
    label = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def to_dict(self):
        return {
            'id': self.id,
            'media_id': self.media_id,
            'strategy_id': self.strategy_id,
            'feature_vector': self.feature_vector,
            'feature_names': self.feature_names,
            'label': self.label
        }
