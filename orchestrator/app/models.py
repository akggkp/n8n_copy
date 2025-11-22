# orchestrator/app/models.py - COMPLETE VERSION
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class MediaItem(Base):
    __tablename__ = "media_items"
    
    id = Column(Integer, primary_key=True)
    video_id = Column(String(255), unique=True, nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size = Column(Integer)
    duration = Column(Float)
    frame_count = Column(Integer)
    status = Column(String(50), default='pending', index=True)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    transcripts = relationship("Transcript", back_populates="media_item", cascade="all, delete-orphan")
    keyword_hits = relationship("KeywordHit", back_populates="media_item", cascade="all, delete-orphan")
    clips = relationship("Clip", back_populates="media_item", cascade="all, delete-orphan")
    embeddings = relationship("Embedding", back_populates="media_item", cascade="all, delete-orphan")
    proven_strategies = relationship("ProvenStrategy", back_populates="media_item", cascade="all, delete-orphan")

class Transcript(Base):
    __tablename__ = "transcripts"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"), nullable=False, index=True)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    media_item = relationship("MediaItem", back_populates="transcripts")

class KeywordHit(Base):
    __tablename__ = "keyword_hits"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"), nullable=False, index=True)
    keyword = Column(String(100), nullable=False, index=True)
    category = Column(String(50), index=True)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float)
    confidence = Column(Float, default=1.0)
    context_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    media_item = relationship("MediaItem", back_populates="keyword_hits")
    clips = relationship("Clip", back_populates="keyword_hit", cascade="all, delete-orphan")

class Clip(Base):
    __tablename__ = "clips"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"), nullable=False, index=True)
    keyword_hit_id = Column(Integer, ForeignKey("keyword_hits.id", ondelete="CASCADE"), index=True)
    file_path = Column(Text, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    media_item = relationship("MediaItem", back_populates="clips")
    keyword_hit = relationship("KeywordHit", back_populates="clips")

class Embedding(Base):
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"), nullable=False, index=True)
    embedding_type = Column(String(50), nullable=False, index=True)  # transcript, frame, clip, keyword
    reference_id = Column(Integer)  # ID of transcript/frame/clip
    embedding_vector = Column(JSON, nullable=False)  # Store as JSON array
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    media_item = relationship("MediaItem", back_populates="embeddings")

class ProvenStrategy(Base):
    __tablename__ = "proven_strategies"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"), nullable=False, index=True)
    strategy_id = Column(Integer)  # From backtesting service
    
    # Performance metrics
    sharpe_ratio = Column(Float)
    win_rate = Column(Float)
    max_drawdown = Column(Float)
    total_return = Column(Float)
    num_trades = Column(Integer)
    avg_trade_duration = Column(Float)
    
    # Status
    status = Column(String(50), default='promoted', index=True)
    promoted_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    media_item = relationship("MediaItem", back_populates="proven_strategies")