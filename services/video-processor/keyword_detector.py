# keyword_detector.py
# Module for detecting trading keywords in transcripts with context windows

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class KeywordCategory(Enum):
    """Trading keyword categories for classification"""
    TECHNICAL_INDICATOR = "technical_indicator"
    PRICE_ACTION = "price_action"
    RISK_MANAGEMENT = "risk_management"
    ORDER_TYPE = "order_type"
    MARKET_CONDITION = "market_condition"
    STRATEGY_TYPE = "strategy_type"
    TIMEFRAME = "timeframe"


@dataclass
class KeywordHit:
    """Represents a detected keyword occurrence"""
    keyword: str
    category: KeywordCategory
    start_time: float
    end_time: float
    confidence: float
    context_text: str
    segment_index: int


class KeywordDetector:
    """Detects trading keywords in transcripts with timestamps"""
    
    # Comprehensive trading keywords dictionary organized by category
    TRADING_KEYWORDS = {
        KeywordCategory.TECHNICAL_INDICATOR: [
            "RSI", "relative strength index",
            "MACD", "moving average convergence divergence",
            "EMA", "exponential moving average",
            "SMA", "simple moving average",
            "Bollinger Bands", "bollinger",
            "Stochastic", "stochastic oscillator",
            "Fibonacci", "fibonacci retracement",
            "pivot", "pivot point",
            "volume", "volume profile",
            "VWAP", "volume weighted average price",
            "ATR", "average true range",
            "ADX", "average directional index",
            "CCI", "commodity channel index",
            "Williams %R", "williams",
            "momentum", "rate of change", "ROC"
        ],
        KeywordCategory.PRICE_ACTION: [
            "support", "resistance",
            "breakout", "break above", "break below",
            "pullback", "retracement",
            "trend", "uptrend", "downtrend", "sideways",
            "consolidation", "congestion",
            "reversal", "reversal pattern",
            "head and shoulders",
            "double top", "double bottom",
            "triple top", "triple bottom",
            "cup and handle", "flag", "pennant",
            "wedge", "triangle",
            "ascending", "descending", "symmetrical"
        ],
        KeywordCategory.RISK_MANAGEMENT: [
            "stop loss", "stop-loss", "SL",
            "take profit", "take-profit", "TP",
            "risk reward", "risk-reward", "R:R",
            "position size", "lot size",
            "drawdown", "maximum drawdown",
            "win rate", "win%", "winning percentage",
            "profit factor", "PF",
            "Sharpe ratio", "Sharpe",
            "volatility", "VIX"
        ],
        KeywordCategory.ORDER_TYPE: [
            "limit order", "limit",
            "market order", "market",
            "stop order", "stop",
            "trailing stop", "trailing",
            "good til canceled", "GTC",
            "fill or kill", "FOK",
            "immediate or cancel", "IOC"
        ],
        KeywordCategory.MARKET_CONDITION: [
            "bull market", "bullish", "bull",
            "bear market", "bearish", "bear",
            "volatility", "low volatility", "high volatility",
            "trending market",
            "ranging market",
            "gap up", "gap down",
            "liquidity", "illiquid"
        ],
        KeywordCategory.STRATEGY_TYPE: [
            "day trading", "day trade", "scalping",
            "swing trading", "swing trade",
            "position trading",
            "mean reversion",
            "momentum trading", "momentum",
            "arbitrage",
            "hedging", "hedge",
            "carry trade",
            "breakout strategy",
            "pullback trading"
        ],
        KeywordCategory.TIMEFRAME: [
            "1 minute", "5 minute", "15 minute", "30 minute",
            "hourly", "4 hour", "daily", "weekly", "monthly",
            "1M", "5M", "15M", "30M", "1H", "4H", "1D", "1W", "1M"
        ]
    }
    
    def __init__(self, case_sensitive: bool = False, min_confidence: float = 0.8):
        """
        Initialize KeywordDetector
        
        Args:
            case_sensitive: Whether to perform case-sensitive matching
            min_confidence: Minimum confidence threshold for keyword hits
        """
        self.case_sensitive = case_sensitive
        self.min_confidence = min_confidence
        self._compile_regex_patterns()
    
    def _compile_regex_patterns(self) -> None:
        """Compile regex patterns for efficient keyword matching"""
        self.keyword_patterns = {}
        flags = 0 if self.case_sensitive else re.IGNORECASE
        
        for category, keywords in self.TRADING_KEYWORDS.items():
            patterns = []
            for keyword in keywords:
                # Create word boundary pattern for whole word matching
                pattern = r'\b' + re.escape(keyword) + r'\b'
                patterns.append(pattern)
            # Combine all patterns for this category with OR
            combined_pattern = '|'.join(patterns)
            self.keyword_patterns[category] = re.compile(combined_pattern, flags)
    
    def detect_keywords_in_transcript(
        self,
        transcript_segments: List[Dict],
        context_window: int = 50
    ) -> List[KeywordHit]:
        """
        Detect trading keywords in transcript segments
        
        Args:
            transcript_segments: List of dicts with 'text', 'start', 'end', 'segment_index'
            context_window: Number of characters to include as context before/after match
        
        Returns:
            List of KeywordHit objects with detection results
        """
        hits = []
        full_text = " ".join([seg['text'] for seg in transcript_segments])
        
        for category, pattern in self.keyword_patterns.items():
            for match in pattern.finditer(full_text):
                keyword = match.group()
                start_pos = match.start()
                end_pos = match.end()
                
                # Extract context
                context_start = max(0, start_pos - context_window)
                context_end = min(len(full_text), end_pos + context_window)
                context_text = full_text[context_start:context_end]
                
                # Find corresponding segment timestamps
                segment_index, start_time, end_time = self._get_timestamp_for_position(
                    full_text, start_pos, end_pos, transcript_segments
                )
                
                hit = KeywordHit(
                    keyword=keyword,
                    category=category,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=self.min_confidence,
                    context_text=context_text.strip(),
                    segment_index=segment_index
                )
                hits.append(hit)
        
        return hits
    
    def _get_timestamp_for_position(
        self,
        full_text: str,
        start_pos: int,
        end_pos: int,
        transcript_segments: List[Dict]
    ) -> Tuple[int, float, float]:
        """
        Map character position in full text to segment timestamps
        
        Args:
            full_text: Concatenated transcript text
            start_pos: Character position of keyword start
            end_pos: Character position of keyword end
            transcript_segments: Original segment data
        
        Returns:
            Tuple of (segment_index, start_time, end_time)
        """
        char_count = 0
        
        for seg_idx, segment in enumerate(transcript_segments):
            segment_text = segment['text']
            segment_length = len(segment_text) + 1  # +1 for space separator
            
            if char_count <= start_pos < char_count + segment_length:
                return (
                    seg_idx,
                    segment['start'],
                    segment['end']
                )
            
            char_count += segment_length
        
        # Fallback to last segment
        return (
            len(transcript_segments) - 1,
            transcript_segments[-1]['start'],
            transcript_segments[-1]['end']
        )
    
    def filter_hits_by_confidence(
        self,
        hits: List[KeywordHit],
        threshold: float
    ) -> List[KeywordHit]:
        """Filter keyword hits by confidence threshold"""
        return [hit for hit in hits if hit.confidence >= threshold]
    
    def get_hits_by_category(
        self,
        hits: List[KeywordHit],
        category: KeywordCategory
    ) -> List[KeywordHit]:
        """Get keyword hits filtered by category"""
        return [hit for hit in hits if hit.category == category]
    
    def get_unique_keywords(self, hits: List[KeywordHit]) -> Dict[str, int]:
        """Get count of unique keywords detected"""
        keyword_counts = {}
        for hit in hits:
            keyword_counts[hit.keyword] = keyword_counts.get(hit.keyword, 0) + 1
        return keyword_counts


def format_hits_for_database(hits: List[KeywordHit]) -> List[Dict]:
    """
    Format KeywordHit objects for database insertion
    
    Args:
        hits: List of KeywordHit objects
    
    Returns:
        List of dicts ready for database insertion
    """
    return [
        {
            'keyword': hit.keyword,
            'category': hit.category.value,
            'start_time': hit.start_time,
            'end_time': hit.end_time,
            'confidence': hit.confidence,
            'context_text': hit.context_text,
            'segment_index': hit.segment_index
        }
        for hit in hits
    ]