# tests/test_feature_engineering.py

from orchestrator.app.feature_engineering import FeatureEngineer
import numpy as np


def test_keyword_features():
    """Test keyword feature extraction"""
    engineer = FeatureEngineer()

    keyword_hits = [{'keyword': 'RSI',
                     'category': 'technical_indicator',
                     'start_time': 10.0,
                     'end_time': 12.0,
                     'confidence': 0.95},
                    {'keyword': 'breakout',
                     'category': 'price_action',
                     'start_time': 25.0,
                     'end_time': 27.0,
                     'confidence': 0.88},
                    {'keyword': 'stop loss',
                     'category': 'risk_management',
                     'start_time': 40.0,
                     'end_time': 42.0,
                     'confidence': 0.92}]

    features = engineer.extract_keyword_features(keyword_hits)

    assert features['technical_indicator_count'] == 1
    assert features['price_action_count'] == 1
    assert features['risk_management_count'] == 1
    assert features['has_rsi'] == 1
    assert features['has_breakout'] == 1
    assert features['has_stop_loss'] == 1
    assert features['avg_confidence'] > 0.8

    print("✓ Keyword feature extraction test passed")


def test_transcript_features():
    """Test transcript feature extraction"""
    engineer = FeatureEngineer()

    transcripts = [
        {'text': 'Today we will discuss the RSI indicator', 'start_time': 0.0, 'end_time': 3.0},
        {'text': 'For example, when RSI is below 30', 'start_time': 3.0, 'end_time': 6.0},
        {'text': 'First, you need to calculate the gains', 'start_time': 6.0, 'end_time': 9.0}
    ]

    features = engineer.extract_transcript_features(transcripts)

    assert features['total_segments'] == 3
    assert features['has_examples'] == 1
    assert features['has_steps'] == 1
    assert features['speaking_rate_wpm'] > 0

    print("✓ Transcript feature extraction test passed")


def test_embedding_features():
    """Test embedding feature extraction"""
    engineer = FeatureEngineer()

    embeddings = [
        np.random.randn(384),
        np.random.randn(384),
        np.random.randn(384)
    ]

    features = engineer.extract_embedding_features(embeddings)

    assert features['embedding_dim'] == 384
    assert features['avg_embedding_norm'] > 0
    assert 0 <= features['embedding_diversity'] <= 1

    print("✓ Embedding feature extraction test passed")


if __name__ == "__main__":
    test_keyword_features()
    test_transcript_features()
    test_embedding_features()
    print("\n✓ All feature engineering tests passed")
