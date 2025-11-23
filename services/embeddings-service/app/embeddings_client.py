# services/embeddings-service/app/embeddings_client.py
# Client for generating and searching embeddings using
# sentence-transformers and Faiss

import numpy as np
import faiss
from typing import List, Dict
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle

logger = logging.getLogger(__name__)


class EmbeddingsClient:
    """Client for generating and searching embeddings using sentence-transformers and Faiss"""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        faiss_index_dir: str = "/data/processed/faiss",
        device: str = "cpu"
    ):
        """
        Initialize embeddings client

        Args:
            model_name: HuggingFace model name (all-MiniLM-L6-v2 is fast and small)
            faiss_index_dir: Directory to store Faiss indices
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.faiss_index_dir = Path(faiss_index_dir)
        self.device = device

        # Create index directory if not exists
        self.faiss_index_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

        # Load or create indices
        self.transcript_index = None
        self.frame_index = None
        self.clip_index = None

        self.transcript_metadata = {}
        self.frame_metadata = {}
        self.clip_metadata = {}

        self._load_indices()

    def _load_indices(self):
        """Load existing Faiss indices or create new ones"""
        try:
            # Load transcript index
            transcript_index_path = self.faiss_index_dir / "transcript_index.faiss"
            if transcript_index_path.exists():
                self.transcript_index = faiss.read_index(
                    str(transcript_index_path))
                with open(self.faiss_index_dir / "transcript_metadata.pkl", "rb") as f:
                    self.transcript_metadata = pickle.load(f)
                logger.info(
                    f"Loaded transcript index with {self.transcript_index.ntotal} vectors")
            else:
                self.transcript_index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info("Created new transcript index")

            # Load frame index
            frame_index_path = self.faiss_index_dir / "frame_index.faiss"
            if frame_index_path.exists():
                self.frame_index = faiss.read_index(str(frame_index_path))
                with open(self.faiss_index_dir / "frame_metadata.pkl", "rb") as f:
                    self.frame_metadata = pickle.load(f)
                logger.info(
                    f"Loaded frame index with {self.frame_index.ntotal} vectors")
            else:
                self.frame_index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info("Created new frame index")

            # Load clip index
            clip_index_path = self.faiss_index_dir / "clip_index.faiss"
            if clip_index_path.exists():
                self.clip_index = faiss.read_index(str(clip_index_path))
                with open(self.faiss_index_dir / "clip_metadata.pkl", "rb") as f:
                    self.clip_metadata = pickle.load(f)
                logger.info(
                    f"Loaded clip index with {self.clip_index.ntotal} vectors")
            else:
                self.clip_index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info("Created new clip index")

        except Exception as e:
            logger.error(f"Error loading indices: {str(e)}")
            raise

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        try:
            embeddings = self.model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise

    def add_transcript_embeddings(
        self,
        media_item_id: int,
        segments: List[Dict]
    ) -> int:
        """
        Add transcript segment embeddings to index

        Args:
            media_item_id: ID of media item
            segments: List of segment dicts with 'id', 'text', 'start_time', 'end_time'

        Returns:
            Number of embeddings added
        """
        try:
            if not segments:
                return 0

            texts = [seg['text'] for seg in segments]
            embeddings = self.encode_text(texts)

            # Add to index
            start_idx = self.transcript_index.ntotal
            self.transcript_index.add(embeddings)

            # Store metadata
            for i, seg in enumerate(segments):
                idx = start_idx + i
                self.transcript_metadata[idx] = {
                    'media_item_id': media_item_id,
                    'segment_id': seg['id'],
                    'start_time': seg['start_time'],
                    'end_time': seg['end_time'],
                    'text': seg['text']
                }

            logger.info(
                f"Added {len(embeddings)} transcript embeddings for media_item {media_item_id}")
            return len(embeddings)

        except Exception as e:
            logger.error(f"Error adding transcript embeddings: {str(e)}")
            raise

    def add_frame_embeddings(
        self,
        media_item_id: int,
        frames: List[Dict]
    ) -> int:
        """
        Add frame embeddings (using CLIP or other vision model)

        Args:
            media_item_id: ID of media item
            frames: List of frame dicts with 'id', 'file_path', 'timestamp'

        Returns:
            Number of embeddings added
        """
        try:
            if not frames:
                return 0

            # For now, use frame file paths as text
            # In production, use CLIP or similar vision model for actual image
            # embeddings
            texts = [
                f"Frame at {f['timestamp']}s from video {media_item_id}" for f in frames]
            embeddings = self.encode_text(texts)

            # Add to index
            start_idx = self.frame_index.ntotal
            self.frame_index.add(embeddings)

            # Store metadata
            for i, frame in enumerate(frames):
                idx = start_idx + i
                self.frame_metadata[idx] = {
                    'media_item_id': media_item_id,
                    'frame_id': frame['id'],
                    'timestamp': frame['timestamp'],
                    'file_path': frame['file_path']
                }

            logger.info(
                f"Added {len(embeddings)} frame embeddings for media_item {media_item_id}")
            return len(embeddings)

        except Exception as e:
            logger.error(f"Error adding frame embeddings: {str(e)}")
            raise

    def add_clip_embeddings(
        self,
        media_item_id: int,
        clips: List[Dict]
    ) -> int:
        """
        Add clip embeddings (combining keyword + context)

        Args:
            media_item_id: ID of media item
            clips: List of clip dicts with 'id', 'keyword', 'context', 'file_path'

        Returns:
            Number of embeddings added
        """
        try:
            if not clips:
                return 0

            # Combine keyword and context for better embeddings
            texts = [
                f"{clip.get('keyword', '')} {clip.get('context', '')}" for clip in clips]
            embeddings = self.encode_text(texts)

            # Add to index
            start_idx = self.clip_index.ntotal
            self.clip_index.add(embeddings)

            # Store metadata
            for i, clip in enumerate(clips):
                idx = start_idx + i
                self.clip_metadata[idx] = {
                    'media_item_id': media_item_id,
                    'clip_id': clip['id'],
                    'keyword': clip.get('keyword'),
                    'file_path': clip.get('file_path'),
                    'start_time': clip.get('start_time'),
                    'end_time': clip.get('end_time')
                }

            logger.info(
                f"Added {len(embeddings)} clip embeddings for media_item {media_item_id}")
            return len(embeddings)

        except Exception as e:
            logger.error(f"Error adding clip embeddings: {str(e)}")
            raise

    def search_transcripts(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """
        Search transcript embeddings using similarity search

        Args:
            query: Search query text
            top_k: Number of top results
            min_similarity: Minimum similarity score (0-1, higher is more similar)

        Returns:
            List of results with similarity scores and metadata
        """
        try:
            if self.transcript_index.ntotal == 0:
                logger.warning("Transcript index is empty")
                return []

            # Encode query
            query_embedding = self.encode_text([query])[0:1]

            # Search
            distances, indices = self.transcript_index.search(
                query_embedding, top_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue

                # Convert L2 distance to similarity score
                similarity = 1 / (1 + dist)

                if similarity >= min_similarity:
                    metadata = self.transcript_metadata.get(int(idx), {})
                    results.append({
                        'index': int(idx),
                        'similarity': float(similarity),
                        'distance': float(dist),
                        'media_item_id': metadata.get('media_item_id'),
                        'segment_id': metadata.get('segment_id'),
                        'start_time': metadata.get('start_time'),
                        'end_time': metadata.get('end_time'),
                        'text': metadata.get('text')
                    })

            logger.info(
                f"Transcript search for '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching transcripts: {str(e)}")
            raise

    def search_frames(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """Search frame embeddings"""
        try:
            if self.frame_index.ntotal == 0:
                logger.warning("Frame index is empty")
                return []

            query_embedding = self.encode_text([query])[0:1]
            distances, indices = self.frame_index.search(
                query_embedding, top_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue

                similarity = 1 / (1 + dist)

                if similarity >= min_similarity:
                    metadata = self.frame_metadata.get(int(idx), {})
                    results.append({
                        'index': int(idx),
                        'similarity': float(similarity),
                        'media_item_id': metadata.get('media_item_id'),
                        'frame_id': metadata.get('frame_id'),
                        'timestamp': metadata.get('timestamp'),
                        'file_path': metadata.get('file_path')
                    })

            logger.info(
                f"Frame search for '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching frames: {str(e)}")
            raise

    def search_clips(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """Search clip embeddings"""
        try:
            if self.clip_index.ntotal == 0:
                logger.warning("Clip index is empty")
                return []

            query_embedding = self.encode_text([query])[0:1]
            distances, indices = self.clip_index.search(query_embedding, top_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue

                similarity = 1 / (1 + dist)

                if similarity >= min_similarity:
                    metadata = self.clip_metadata.get(int(idx), {})
                    results.append({
                        'index': int(idx),
                        'similarity': float(similarity),
                        'media_item_id': metadata.get('media_item_id'),
                        'clip_id': metadata.get('clip_id'),
                        'keyword': metadata.get('keyword'),
                        'file_path': metadata.get('file_path'),
                        'start_time': metadata.get('start_time'),
                        'end_time': metadata.get('end_time')
                    })

            logger.info(
                f"Clip search for '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching clips: {str(e)}")
            raise

    def save_indices(self):
        """Save indices and metadata to disk"""
        try:
            # Save transcript index
            if self.transcript_index:
                faiss.write_index(
                    self.transcript_index, str(
                        self.faiss_index_dir / "transcript_index.faiss"))
                with open(self.faiss_index_dir / "transcript_metadata.pkl", "wb") as f:
                    pickle.dump(self.transcript_metadata, f)

            # Save frame index
            if self.frame_index:
                faiss.write_index(
                    self.frame_index, str(
                        self.faiss_index_dir / "frame_index.faiss"))
                with open(self.faiss_index_dir / "frame_metadata.pkl", "wb") as f:
                    pickle.dump(self.frame_metadata, f)

            # Save clip index
            if self.clip_index:
                faiss.write_index(
                    self.clip_index, str(
                        self.faiss_index_dir / "clip_index.faiss"))
                with open(self.faiss_index_dir / "clip_metadata.pkl", "wb") as f:
                    pickle.dump(self.clip_metadata, f)

            logger.info("Indices and metadata saved successfully")

        except Exception as e:
            logger.error(f"Error saving indices: {str(e)}")
            raise

    def get_stats(self) -> Dict:
        """Get statistics about loaded indices"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'transcript_vectors': self.transcript_index.ntotal if self.transcript_index else 0,
            'frame_vectors': self.frame_index.ntotal if self.frame_index else 0,
            'clip_vectors': self.clip_index.ntotal if self.clip_index else 0,
            'total_vectors': (
                self.transcript_index.ntotal if self.transcript_index else 0) + (
                self.frame_index.ntotal if self.frame_index else 0) + (
                self.clip_index.ntotal if self.clip_index else 0)}
