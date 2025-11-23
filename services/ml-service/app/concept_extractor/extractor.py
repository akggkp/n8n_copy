"""Trading Concept Extractor - Extract trading concepts from transcription"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """Extract trading concepts from video transcription"""

    def __init__(self):
        self.trading_indicators = {
            'rsi': ['RSI', 'relative strength index'],
            'macd': ['MACD', 'moving average convergence divergence'],
            'ema': ['EMA', 'exponential moving average'],
            'sma': ['SMA', 'simple moving average'],
            'bollinger': ['Bollinger', 'bollinger bands'],
            'stochastic': ['Stochastic', 'stoch'],
            'atr': ['ATR', 'average true range'],
            'adx': ['ADX', 'average directional index'],
            'roc': ['ROC', 'rate of change'],
            'obv': ['OBV', 'on balance volume']
        }

        self.patterns = {
            'head_shoulders': ['head and shoulders', 'head & shoulders', 'HS pattern'],
            'double_top': ['double top', 'double peak'],
            'double_bottom': ['double bottom', 'double trough'],
            'cup_handle': ['cup and handle', 'cup & handle'],
            'flag': ['flag pattern', 'continuation flag'],
            'triangle': ['triangle', 'symmetrical triangle'],
            'wedge': ['wedge'],
            'pennant': ['pennant']
        }

    def extract_trading_concepts(self, transcription: str) -> List[str]:
        """Extract trading concepts"""
        concepts = []
        keywords = [
            'trend', 'momentum', 'reversal', 'breakout', 'pullback',
            'support', 'resistance', 'pivot', 'bullish', 'bearish',
            'consolidation', 'divergence', 'convergence', 'oversold',
            'overbought', 'entry', 'exit', 'stop loss', 'take profit',
            'risk management', 'position sizing', 'volatility'
        ]

        text_lower = transcription.lower()
        for concept in keywords:
            if concept in text_lower:
                concepts.append(concept)

        return list(set(concepts))  # Remove duplicates

    def extract_indicators(self, transcription: str) -> List[str]:
        """Extract technical indicators mentioned"""
        indicators = []
        text_lower = transcription.lower()

        for indicator_key, keywords in self.trading_indicators.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    indicators.append(indicator_key.upper())
                    break

        return list(set(indicators))  # Remove duplicates

    def extract_patterns(
            self,
            transcription: str,
            detected_charts: List[Dict] = None) -> List[str]:
        """Extract chart patterns mentioned"""
        patterns = []
        text_lower = transcription.lower()

        for pattern_key, keywords in self.patterns.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    patterns.append(pattern_key)
                    break

        return list(set(patterns))  # Remove duplicates
