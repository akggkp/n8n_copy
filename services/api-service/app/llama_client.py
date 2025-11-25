# services/api-service/app/llama_client.py
# Client for interacting with Ollama API for Llama inference

import requests
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class LlamaClient:
    """Client for generating trading strategies using Llama via Ollama"""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize Llama client

        Args:
            ollama_url: Base URL for Ollama API (default: local)
        """
        self.ollama_url = ollama_url
        self.model = "llama2"

    def generate_strategy(
        self,
        examples: List[Dict],
        keyword: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate trading strategy from examples

        Args:
            examples: List of example dicts with transcript, keyword, concepts
            keyword: Keyword to focus on (e.g., "RSI")
            temperature: Generation temperature (0.0-1.0, higher = more creative)
            max_tokens: Maximum tokens in response

        Returns:
            Generated strategy text
        """
        try:
            # Build context from examples
            context_lines = []
            for ex in examples[:5]:  # Use top 5 examples
                transcript = ex.get('transcript', '')[:200]
                concepts = ", ".join(ex.get('detected_concepts', []))
                context_lines.append(
                    f"Timestamp {ex.get('timestamp', 0):.1f}s: {transcript}")
                if concepts:
                    context_lines.append(f"  Concepts: {concepts}")

            context = "\n".join(context_lines)

            prompt = (
                f"Based on these trading video extracts about {keyword}:\n\n{context}\n\n"
                f"Generate a concise trading strategy that uses {keyword}. Include:\n"
                "1. Entry signals\n"
                "2. Exit criteria\n"
                "3. Risk management rules\n"
                "4. Expected outcomes\n\n"
                "Strategy:"
            )

            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature
                },
                timeout=120
            )

            if response.status_code != 200:
                logger.error(f"Ollama error: {response.text}")
                return None

            result = response.json()
            strategy_text = result.get('response', '')

            logger.info(
                f"Strategy generated for {keyword} ({len(strategy_text)} chars)")
            return strategy_text

        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return None

    def summarize_keyword(
        self,
        examples: List[Dict],
        keyword: str
    ) -> str:
        """
        Summarize trading concept for a keyword

        Args:
            examples: List of example dicts
            keyword: Keyword to summarize

        Returns:
            Summary text
        """
        try:
            # Combine all transcripts
            all_text = " ".join([ex.get('transcript', '')
                                for ex in examples[:10]])

            prompt = (
                f"Summarize the key concepts and trading insights about {keyword} "
                f"based on:\n\n{all_text[:1000]}\n\nSummary (3-5 sentences):"
            )

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.5
                },
                timeout=60
            )

            if response.status_code != 200:
                logger.error(f"Ollama error: {response.text}")
                return None

            result = response.json()
            summary = result.get('response', '')

            logger.info(f"Summary generated for {keyword}")
            return summary

        except Exception as e:
            logger.error(f"Error summarizing keyword: {str(e)}")
            return None

    def health_check(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {str(e)}")
            return False