# services/api-service/app/schemas.py
# Pydantic models for request/response validation

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class MediaStatusEnum(str, Enum):
    """Status of media processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class KeywordCategoryEnum(str, Enum):
    """Keyword categories"""
    TECHNICAL_INDICATOR = "technical_indicator"
    PRICE_ACTION = "price_action"
    RISK_MANAGEMENT = "risk_management"
    ORDER_TYPE = "order_type"
    MARKET_CONDITION = "market_condition"
    STRATEGY_TYPE = "strategy_type"
    TIMEFRAME = "timeframe"


class EmbeddingTypeEnum(str, Enum):
    """Embedding types"""
    TRANSCRIPT = "transcript"
    FRAME = "frame"
    CLIP = "clip"


# ============================================================================
# MEDIA ITEM SCHEMAS
# ============================================================================

class MediaItemBase(BaseModel):
    """Base media item schema"""
    video_id: str
    filename: str
    source_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None


class MediaItemCreate(MediaItemBase):
    """Schema for creating media item"""


class MediaItemUpdate(BaseModel):
    """Schema for updating media item"""
    status: Optional[MediaStatusEnum] = None
    duration_seconds: Optional[float] = None


class MediaItemResponse(MediaItemBase):
    """Schema for media item response"""
    id: int
    status: MediaStatusEnum
    created_at: datetime
    updated_at: datetime
    clips_count: int = 0
    keywords_count: int = 0
    transcript_segments_count: int = 0

    class Config:
        from_attributes = True


class MediaItemListResponse(BaseModel):
    """Response for listing media items"""
    total: int
    skip: int
    limit: int
    items: List[MediaItemResponse]


# ============================================================================
# TRANSCRIPT SCHEMAS
# ============================================================================

class TranscriptSegmentBase(BaseModel):
    """Base transcript segment schema"""
    segment_index: int
    start_time: float
    end_time: float
    text: str
    language: Optional[str] = "en"


class TranscriptSegmentCreate(TranscriptSegmentBase):
    """Schema for creating transcript segment"""


class TranscriptSegmentResponse(TranscriptSegmentBase):
    """Response schema for transcript segment"""
    id: int
    media_item_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class TranscriptResponse(BaseModel):
    """Response for full transcript"""
    media_id: int
    segments: List[TranscriptSegmentResponse]
    total_segments: int


# ============================================================================
# KEYWORD SCHEMAS
# ============================================================================

class KeywordHitBase(BaseModel):
    """Base keyword hit schema"""
    keyword: str
    start_time: float
    end_time: float
    confidence: float = Field(ge=0.0, le=1.0)
    context_text: Optional[str] = None


class KeywordHitCreate(KeywordHitBase):
    """Schema for creating keyword hit"""
    category: KeywordCategoryEnum


class KeywordHitResponse(KeywordHitBase):
    """Response schema for keyword hit"""
    id: int
    media_item_id: int
    category: KeywordCategoryEnum
    created_at: datetime

    class Config:
        from_attributes = True


class KeywordListResponse(BaseModel):
    """Response for keyword list"""
    keywords: List[KeywordHitResponse]
    total: int
    media_id: Optional[int] = None


# ============================================================================
# CLIP SCHEMAS
# ============================================================================

class ClipBase(BaseModel):
    """Base clip schema"""
    keyword: str
    start_time: float
    end_time: float
    duration_seconds: float
    file_path: str
    file_size_bytes: Optional[int] = None


class ClipCreate(ClipBase):
    """Schema for creating clip"""


class ClipResponse(ClipBase):
    """Response schema for clip"""
    id: int
    media_item_id: int
    keyword_hit_id: Optional[int] = None
    created_at: datetime
    download_url: str = Field(default="")

    class Config:
        from_attributes = True

    @validator("download_url", pre=True, always=True)
    def generate_download_url(cls, v, values):
        """Generate download URL from clip ID"""
        if "id" in values:
            return f"/clip/{values['id']}/download"
        return v


class ClipListResponse(BaseModel):
    """Response for clip list"""
    total: int
    skip: int
    limit: int
    clips: List[ClipResponse]


# ============================================================================
# FRAME SCHEMAS
# ============================================================================

class FrameBase(BaseModel):
    """Base frame schema"""
    timestamp: float
    file_path: str
    width: Optional[int] = None
    height: Optional[int] = None
    contains_chart: bool = False


class FrameCreate(FrameBase):
    """Schema for creating frame"""


class FrameResponse(FrameBase):
    """Response schema for frame"""
    id: int
    media_item_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# EMBEDDING SCHEMAS
# ============================================================================

class EmbeddingBase(BaseModel):
    """Base embedding schema"""
    embedding_type: EmbeddingTypeEnum
    reference_id: Optional[int] = None
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_dimension: int = 384


class EmbeddingCreate(EmbeddingBase):
    """Schema for creating embedding"""
    embedding_vector: List[float]


class EmbeddingResponse(EmbeddingBase):
    """Response schema for embedding (without full vector)"""
    id: int
    media_item_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class EmbeddingSearchResult(BaseModel):
    """Result from embedding similarity search"""
    result_id: int
    reference_type: EmbeddingTypeEnum
    reference_id: int
    similarity_score: float
    metadata: Dict[str, Any] = {}


class EmbeddingSearchResponse(BaseModel):
    """Response for embedding search"""
    query: str
    top_k: int
    results: List[EmbeddingSearchResult]


# ============================================================================
# LLAMA DATASET SCHEMAS
# ============================================================================

class LlamaExampleBase(BaseModel):
    """Base Llama training example"""
    clip_id: str
    transcript: str
    keyword: str
    category: Optional[KeywordCategoryEnum] = None
    timestamp: float
    clip_url: Optional[str] = None
    frame_path: Optional[str] = None
    context_text: Optional[str] = None
    detected_concepts: List[str] = []


class LlamaExampleWithEmbeddings(LlamaExampleBase):
    """Llama example with embeddings"""
    embeddings: Optional[List[float]] = None


class LlamaExamplesResponse(BaseModel):
    """Response for Llama examples"""
    examples: List[LlamaExampleWithEmbeddings]
    total: int
    keyword_filter: Optional[str] = None
    category_filter: Optional[str] = None


# ============================================================================
# INGEST/PIPELINE SCHEMAS
# ============================================================================

class IngestRequest(BaseModel):
    """Request schema for video ingest"""
    video_path: str
    source_url: Optional[str] = None
    filename: Optional[str] = None


class IngestResponse(BaseModel):
    """Response schema for ingest"""
    status: str
    task_id: str
    media_item_id: int
    message: str
    estimated_duration_minutes: int


class TaskStatusResponse(BaseModel):
    """Response for task status"""
    task_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress: Optional[float] = None
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# ============================================================================
# SEARCH/QUERY SCHEMAS
# ============================================================================

class MediaItemQuery(BaseModel):
    """Query parameters for media item search"""
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)
    status: Optional[MediaStatusEnum] = None


class ClipQuery(BaseModel):
    """Query parameters for clip search"""
    keyword: Optional[str] = None
    media_id: Optional[int] = None
    limit: int = Field(default=10, ge=1, le=100)
    skip: int = Field(default=0, ge=0)


class KeywordQuery(BaseModel):
    """Query parameters for keyword search"""
    media_id: Optional[int] = None
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    limit: int = Field(default=50, ge=1, le=500)
    category: Optional[KeywordCategoryEnum] = None


class EmbeddingSearchQuery(BaseModel):
    """Query parameters for embedding search"""
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    embedding_type: EmbeddingTypeEnum = EmbeddingTypeEnum.TRANSCRIPT
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)


class LlamaExampleQuery(BaseModel):
    """Query parameters for Llama examples"""
    keyword: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=50)
    include_embeddings: bool = False
    category: Optional[KeywordCategoryEnum] = None


# ============================================================================
# ERROR SCHEMAS
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: str
    timestamp: datetime
    request_id: Optional[str] = None


class ValidationErrorResponse(BaseModel):
    """Validation error response"""
    error: str = "Validation error"
    details: List[Dict[str, Any]]
    timestamp: datetime
