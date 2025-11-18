CREATE DATABASE trading_education;

CREATE TABLE IF NOT EXISTS processed_videos (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(255) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    status VARCHAR(50),
    processing_stats JSONB DEFAULT NULL,
    processing_time_seconds NUMERIC,
    processed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS video_frames (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES processed_videos(id),
    frame_index INTEGER,
    detected_charts JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
