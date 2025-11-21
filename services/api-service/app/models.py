# services/api-service/app/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class MediaItem(Base):
    __tablename__ = "media_items"
    
    id = Column(Integer, primary_key=True)
    video_id = Column(String(255), unique=True, nullable=False)
    source_url = Column(Text)
    filename = Column(String(255), nullable=False)
    duration_seconds = Column(Float)
    file_size_bytes = Column(Integer)
    status = Column(String(50), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    transcripts = relationship("Transcript", back_populates="media_item", cascade="all, delete-orphan")
    keyword_hits = relationship("KeywordHit", back_populates="media_item", cascade="all, delete-orphan")
    frames = relationship("Frame", back_populates="media_item", cascade="all, delete-orphan")
    clips = relationship("Clip", back_populates="media_item", cascade="all, delete-orphan")
    embeddings = relationship("Embedding", back_populates="media_item", cascade="all, delete-orphan")

class Transcript(Base):
    __tablename__ = "transcripts"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"))
    segment_index = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    language = Column(String(10), default="en")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    media_item = relationship("MediaItem", back_populates="transcripts")

class KeywordHit(Base):
    __tablename__ = "keyword_hits"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"))
    keyword = Column(String(100), nullable=False)
    category = Column(String(50))
    start_time = Column(Float, nullable=False)
    end_time = Column(Float)
    confidence = Column(Float, default=1.0)
    context_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    media_item = relationship("MediaItem", back_populates="keyword_hits")
    clips = relationship("Clip", back_populates="keyword_hit")

class Clip(Base):
    __tablename__ = "clips"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"))
    keyword_hit_id = Column(Integer, ForeignKey("keyword_hits.id", ondelete="SET NULL"))
    keyword = Column(String(100))
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration_seconds = Column(Float)
    file_path = Column(Text, nullable=False)
    file_size_bytes = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    media_item = relationship("MediaItem", back_populates="clips")
    keyword_hit = relationship("KeywordHit", back_populates="clips")

class Frame(Base):
    __tablename__ = "frames"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"))
    timestamp = Column(Float, nullable=False)
    file_path = Column(Text, nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    contains_chart = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    media_item = relationship("MediaItem", back_populates="frames")

class Embedding(Base):
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"))
    embedding_type = Column(String(50))
    reference_id = Column(Integer)
    embedding_model = Column(String(100))
    embedding_vector = Column(ARRAY(Float))
    vector_dimension = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    media_item = relationship("MediaItem", back_populates="embeddings")

class StrategiesFeature(Base):
    __tablename__ = "strategies_features"
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("proven_strategies.id", ondelete="CASCADE"))
    media_item_id = Column(Integer, ForeignKey("media_items.id", ondelete="SET NULL"))
    feature_name = Column(String(255), nullable=False)
    feature_value = Column(Float, nullable=False)
    feature_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
