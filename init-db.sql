CREATE TABLE IF NOT EXISTS video_frames (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES processed_videos(id),
    frame_index INTEGER,
    detected_charts JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS proven_strategies (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(255) UNIQUE NOT NULL,
    strategy_name VARCHAR(255) NOT NULL,
    strategy_data TEXT,
    backtest_results TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ml_strategies (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(255) UNIQUE NOT NULL,
    strategy_data TEXT,
    is_profitable BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(255) NOT NULL,
    win_rate NUMERIC(5,2),
    profit_factor NUMERIC(10,2),
    sharpe_ratio NUMERIC(10,2),
    total_pnl NUMERIC(15,2),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    tested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_video_id ON processed_videos(video_id);
CREATE INDEX IF NOT EXISTS idx_status ON processed_videos(status);
CREATE INDEX IF NOT EXISTS idx_proven_strategies_video ON proven_strategies(video_id);