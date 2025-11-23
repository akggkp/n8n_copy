# orchestrator/app/feature_engineering.py
# Extract features from processed videos for ML/RL training

import numpy as np
from typing import List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extract trading strategy features from video processing results"""

    def __init__(self):
        self.feature_names = []

    def extract_keyword_features(self, keyword_hits: List[Dict]) -> Dict:
        """
        Extract features from keyword detection results

        Args:
            keyword_hits: List of keyword hit dicts with timestamp, confidence, category

        Returns:
            Feature dict with counts, frequencies, confidence stats
        """
        try:
            if not keyword_hits:
                return self._empty_keyword_features()

            features = {}

            # Category counts
            categories = [hit.get('category', 'unknown')
                          for hit in keyword_hits]
            features['technical_indicator_count'] = categories.count(
                'technical_indicator')
            features['price_action_count'] = categories.count('price_action')
            features['risk_management_count'] = categories.count(
                'risk_management')
            features['order_type_count'] = categories.count('order_type')

            # Keyword diversity
            unique_keywords = len(
                set([hit.get('keyword', '') for hit in keyword_hits]))
            features['keyword_diversity'] = unique_keywords / \
                len(keyword_hits) if keyword_hits else 0.0

            # Confidence statistics
            confidences = [hit.get('confidence', 0.0) for hit in keyword_hits]
            features['avg_confidence'] = np.mean(
                confidences) if confidences else 0.0
            features['min_confidence'] = np.min(
                confidences) if confidences else 0.0
            features['max_confidence'] = np.max(
                confidences) if confidences else 0.0

            # Temporal features
            timestamps = sorted([hit.get('start_time', 0.0)
                                for hit in keyword_hits])
            if len(timestamps) > 1:
                features['keyword_frequency'] = len(
                    timestamps) / (timestamps[-1] - timestamps[0] + 1)
                features['avg_time_between_keywords'] = np.mean(
                    np.diff(timestamps))
            else:
                features['keyword_frequency'] = 0.0
                features['avg_time_between_keywords'] = 0.0

            # Binary flags for important concepts
            keywords_lower = [hit.get('keyword', '').lower()
                              for hit in keyword_hits]
            features['has_rsi'] = 1 if 'rsi' in keywords_lower else 0
            features['has_macd'] = 1 if 'macd' in keywords_lower else 0
            features['has_support_resistance'] = 1 if any(
                k in keywords_lower for k in ['support', 'resistance']) else 0
            features['has_breakout'] = 1 if 'breakout' in keywords_lower else 0
            features['has_stop_loss'] = 1 if 'stop loss' in keywords_lower or 'stop-loss' in keywords_lower else 0

            return features

        except Exception as e:
            logger.error(f"Error extracting keyword features: {str(e)}")
            return self._empty_keyword_features()

    def extract_transcript_features(self, transcripts: List[Dict]) -> Dict:
        """
        Extract features from transcript analysis

        Args:
            transcripts: List of transcript segment dicts with text, timestamps

        Returns:
            Feature dict with text statistics
        """
        try:
            if not transcripts:
                return self._empty_transcript_features()

            features = {}

            # Text length statistics
            texts = [t.get('text', '') for t in transcripts]
            text_lengths = [len(text.split()) for text in texts]

            features['total_words'] = sum(text_lengths)
            features['avg_words_per_segment'] = np.mean(
                text_lengths) if text_lengths else 0.0
            features['total_segments'] = len(transcripts)

            # Speaking rate (words per minute)
            if transcripts and len(transcripts) > 1:
                duration = transcripts[-1].get('end_time', 0) - \
                    transcripts[0].get('start_time', 0)
                features['speaking_rate_wpm'] = (
                    features['total_words'] / duration) * 60 if duration > 0 else 0.0
            else:
                features['speaking_rate_wpm'] = 0.0

            # Educational signal detection (simple heuristics)
            full_text = " ".join(texts).lower()

            features['has_explanation'] = 1 if any(
                word in full_text for word in [
                    'because', 'therefore', 'reason', 'due to']) else 0
            features['has_examples'] = 1 if any(
                word in full_text for word in [
                    'example', 'for instance', 'such as']) else 0
            features['has_steps'] = 1 if any(
                word in full_text for word in [
                    'first', 'second', 'third', 'step']) else 0
            features['question_density'] = full_text.count(
                '?') / len(texts) if texts else 0.0

            return features

        except Exception as e:
            logger.error(f"Error extracting transcript features: {str(e)}")
            return self._empty_transcript_features()

    def extract_embedding_features(self, embeddings: List[np.ndarray]) -> Dict:
        """
        Extract features from embedding vectors

        Args:
            embeddings: List of embedding vectors (numpy arrays)

        Returns:
            Feature dict with embedding statistics
        """
        try:
            if not embeddings or len(embeddings) == 0:
                return self._empty_embedding_features()

            features = {}

            # Stack embeddings
            emb_matrix = np.vstack(embeddings)

            # Embedding statistics
            features['embedding_dim'] = emb_matrix.shape[1]
            features['avg_embedding_norm'] = np.mean(
                np.linalg.norm(emb_matrix, axis=1))

            # Diversity: average pairwise cosine similarity
            if len(embeddings) > 1:
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        )
                        similarities.append(sim)

                features['avg_embedding_similarity'] = np.mean(
                    similarities) if similarities else 0.0
                features['embedding_diversity'] = 1.0 - \
                    features['avg_embedding_similarity']
            else:
                features['avg_embedding_similarity'] = 0.0
                features['embedding_diversity'] = 0.0

            return features

        except Exception as e:
            logger.error(f"Error extracting embedding features: {str(e)}")
            return self._empty_embedding_features()

    def build_feature_vector(
        self,
        keyword_hits: List[Dict],
        transcripts: List[Dict],
        embeddings: Optional[List[np.ndarray]] = None
    ) -> Dict:
        """
        Build complete feature vector for a media item

        Args:
            keyword_hits: Keyword detection results
            transcripts: Transcript segments
            embeddings: Optional embedding vectors

        Returns:
            Complete feature dict
        """
        try:
            features = {}

            # Extract features from each source
            keyword_features = self.extract_keyword_features(keyword_hits)
            transcript_features = self.extract_transcript_features(transcripts)

            if embeddings:
                embedding_features = self.extract_embedding_features(
                    embeddings)
            else:
                embedding_features = self._empty_embedding_features()

            # Combine all features
            features.update(keyword_features)
            features.update(transcript_features)
            features.update(embedding_features)

            # Add metadata
            features['timestamp'] = datetime.utcnow().isoformat()
            features['feature_count'] = len(features)

            # Store feature names for later use
            self.feature_names = list(features.keys())

            logger.info(f"Built feature vector with {len(features)} features")
            return features

        except Exception as e:
            logger.error(f"Error building feature vector: {str(e)}")
            return {}

    def features_to_array(self, features: Dict) -> np.ndarray:
        """Convert feature dict to numpy array for ML models"""
        try:
            # Exclude non-numeric fields
            numeric_features = {
                k: v for k, v in features.items() if isinstance(
                    v, (int, float))}
            return np.array(list(numeric_features.values()))
        except Exception as e:
            logger.error(f"Error converting features to array: {str(e)}")
            return np.array([])

    def _empty_keyword_features(self) -> Dict:
        """Return empty keyword feature dict"""
        return {
            'technical_indicator_count': 0,
            'price_action_count': 0,
            'risk_management_count': 0,
            'order_type_count': 0,
            'keyword_diversity': 0.0,
            'avg_confidence': 0.0,
            'min_confidence': 0.0,
            'max_confidence': 0.0,
            'keyword_frequency': 0.0,
            'avg_time_between_keywords': 0.0,
            'has_rsi': 0,
            'has_macd': 0,
            'has_support_resistance': 0,
            'has_breakout': 0,
            'has_stop_loss': 0
        }

    def _empty_transcript_features(self) -> Dict:
        """Return empty transcript feature dict"""
        return {
            'total_words': 0,
            'avg_words_per_segment': 0.0,
            'total_segments': 0,
            'speaking_rate_wpm': 0.0,
            'has_explanation': 0,
            'has_examples': 0,
            'has_steps': 0,
            'question_density': 0.0
        }

    def _empty_embedding_features(self) -> Dict:
        """Return empty embedding feature dict"""
        return {
            'embedding_dim': 0,
            'avg_embedding_norm': 0.0,
            'avg_embedding_similarity': 0.0,
            'embedding_diversity': 0.0
        }
