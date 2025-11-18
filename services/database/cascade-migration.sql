DROP VIEW IF EXISTS cascade_statistics CASCADE;

CREATE VIEW cascade_statistics AS
SELECT
    pv.video_id,
    pv.filename,
    COALESCE(pv.processing_time_seconds, 0)::NUMERIC(10,2) as processing_time_seconds,
    (pv.processing_stats->>'total_frames')::INTEGER as total_frames,
    (pv.processing_stats->>'nano_only')::INTEGER as nano_only_detections,
    (pv.processing_stats->>'cascade_used')::INTEGER as cascade_refinements,
    (pv.processing_stats->>'total_detections')::INTEGER as total_detections,
    ROUND(
        CAST(
            ((pv.processing_stats->>'cascade_used')::FLOAT /
             NULLIF((pv.processing_stats->>'total_frames')::FLOAT, 0)) * 100
        AS NUMERIC),
        1
    ) as cascade_percentage,
    COALESCE(pv.processed_at, pv.created_at) as processed_at
FROM processed_videos pv
WHERE pv.processing_stats IS NOT NULL
ORDER BY pv.processed_at DESC;
