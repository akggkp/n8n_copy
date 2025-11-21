-- init-db-extensions.sql
-- Database schema extensions for Trading Media Extraction Pipeline
-- Add new tables for media items, transcripts, keywords, frames, clips, embeddings, and ML features

-- Table for core video metadata and source URL support
CREATE TABLE IF NOT EXISTS media_items (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(255) UNIQUE NOT NULL,
    source_url TEXT,
    filename VARCHAR(255) NOT NULL,
    duration_seconds FLOAT,
    file_size_bytes BIGINT,
    status VARCHAR(50) DEFAULT 'pending', -- 'processing', 'completed', 'failed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Timestamped transcript segments from Whisper (word or segment-level)
CREATE TABLE IF NOT EXISTS transcripts (
    id SERIAL PRIMARY KEY,
    media_item_id INT NOT NULL REFERENCES media_items(id) ON DELETE CASCADE,
    segment_index INT NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    text TEXT NOT NULL,
    language VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_media_transcript FOREIGN KEY(media_item_id) REFERENCES media_items(id)
);

-- Keyword occurrences with timestamps and confidence scores
CREATE TABLE IF NOT EXISTS keyword_hits (
    id SERIAL PRIMARY KEY,
    media_item_id INT NOT NULL REFERENCES media_items(id) ON DELETE CASCADE,
    keyword VARCHAR(100) NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT,
    confidence FLOAT DEFAULT 1.0,
    context_text TEXT, -- surrounding text for context
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_media_keyword FOREIGN KEY(media_item_id) REFERENCES media_items(id)
);

-- Extracted frames linked to timestamps with storage paths
CREATE TABLE IF NOT EXISTS frames (
    id SERIAL PRIMARY KEY,
    media_item_id INT NOT NULL REFERENCES media_items(id) ON DELETE CASCADE,
    timestamp FLOAT NOT NULL,
    file_path TEXT NOT NULL,
    width INT,
    height INT,
    contains_chart BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_media_frame FOREIGN KEY(media_item_id) REFERENCES media_items(id)
);

-- Video clips generated around keyword hits with paths and duration
CREATE TABLE IF NOT EXISTS clips (
    id SERIAL PRIMARY KEY,
    media_item_id INT NOT NULL REFERENCES media_items(id) ON DELETE CASCADE,
    keyword_hit_id INT REFERENCES keyword_hits(id) ON DELETE SET NULL,
    keyword VARCHAR(100),
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    duration_seconds FLOAT,
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_media_clip FOREIGN KEY(media_item_id) REFERENCES media_items(id)
);

-- Embeddings storage for transcripts and frames (text/image vectors)
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    media_item_id INT NOT NULL REFERENCES media_items(id) ON DELETE CASCADE,
    embedding_type VARCHAR(50) NOT NULL, -- 'transcript', 'frame', 'clip'
    reference_id INT, -- transcript id, frame id, or clip id
    embedding_model VARCHAR(100), -- model used (e.g., 'all-MiniLM-L6-v2')
    embedding_vector FLOAT8[] NOT NULL,
    vector_dimension INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_media_embedding FOREIGN KEY(media_item_id) REFERENCES media_items(id)
);

-- Numeric features extracted from signals for ML/RL training
CREATE TABLE IF NOT EXISTS strategies_features (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES proven_strategies(id) ON DELETE CASCADE,
    media_item_id INT REFERENCES media_items(id) ON DELETE SET NULL,
    feature_name VARCHAR(255) NOT NULL,
    feature_value FLOAT NOT NULL,
    feature_type VARCHAR(50), -- 'keyword_count', 'chart_pattern', 'sentiment', etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_strategy_features FOREIGN KEY(strategy_id) REFERENCES proven_strategies(id)
);

-- ML/RL model predictions and results
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    media_item_id INT NOT NULL REFERENCES media_items(id) ON DELETE CASCADE,
    strategy_id INT NOT NULL REFERENCES proven_strategies(id) ON DELETE CASCADE,
    predicted_action VARCHAR(50), -- 'buy', 'sell', 'hold'
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Backtest results for ML-generated strategies
CREATE TABLE IF NOT EXISTS ml_backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES proven_strategies(id) ON DELETE CASCADE,
    sharpe_ratio FLOAT,
    profit_factor FLOAT,
    win_rate FLOAT,
    total_return FLOAT,
    max_drawdown FLOAT,
    trades_count INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices for performance optimization
CREATE INDEX IF NOT EXISTS idx_media_items_video_id ON media_items(video_id);
CREATE INDEX IF NOT EXISTS idx_media_items_status ON media_items(status);
CREATE INDEX IF NOT EXISTS idx_transcripts_media_id ON transcripts(media_item_id);
CREATE INDEX IF NOT EXISTS idx_keyword_hits_media_id ON keyword_hits(media_item_id);
CREATE INDEX IF NOT EXISTS idx_keyword_hits_keyword ON keyword_hits(keyword);
CREATE INDEX IF NOT EXISTS idx_frames_media_id ON frames(media_item_id);
CREATE INDEX IF NOT EXISTS idx_clips_media_id ON clips(media_item_id);
CREATE INDEX IF NOT EXISTS idx_clips_keyword ON clips(keyword);
CREATE INDEX IF NOT EXISTS idx_embeddings_media_id ON embeddings(media_item_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings(embedding_type);
CREATE INDEX IF NOT EXISTS idx_strategies_features_strategy_id ON strategies_features(strategy_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_strategy_id ON model_predictions(strategy_id);
CREATE INDEX IF NOT EXISTS idx_ml_backtest_results_strategy_id ON ml_backtest_results(strategy_id);