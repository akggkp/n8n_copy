# tests/performance/benchmark.py
# Benchmark suite for key operations

import time
import numpy as np
from orchestrator.app.feature_engineering import FeatureEngineer


class Benchmark:
    """Benchmarking utility"""

    def __init__(self, name: str):
        self.name = name
        self.times = []

    def run(self, func, *args, iterations=10, **kwargs):
        """Run function multiple times and record timings"""
        print(f"\nBenchmark: {self.name}")
        print(f"Iterations: {iterations}")

        for i in range(iterations):
            start = time.time()
            func(*args, **kwargs)
            duration = time.time() - start
            self.times.append(duration)

        avg = np.mean(self.times)
        std = np.std(self.times)
        min_time = np.min(self.times)
        max_time = np.max(self.times)

        print("Results:")
        print(f"  Average: {avg*1000:.2f}ms")
        print(f"  Std Dev: {std*1000:.2f}ms")
        print(f"  Min: {min_time*1000:.2f}ms")
        print(f"  Max: {max_time*1000:.2f}ms")

        return {
            "avg": avg,
            "std": std,
            "min": min_time,
            "max": max_time
        }


def benchmark_feature_extraction():
    """Benchmark feature extraction performance"""
    engineer = FeatureEngineer()

    # Sample data
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
                     'confidence': 0.92}] * 10  # 30 hits

    transcripts = [
        {'text': 'Today we will discuss trading', 'start_time': 0.0, 'end_time': 3.0},
        {'text': 'RSI is a momentum indicator', 'start_time': 3.0, 'end_time': 6.0}
    ] * 20  # 40 segments

    embeddings = [np.random.randn(384) for _ in range(50)]

    # Benchmark keyword extraction
    bench1 = Benchmark("Keyword Feature Extraction")
    bench1.run(engineer.extract_keyword_features, keyword_hits, iterations=100)

    # Benchmark transcript extraction
    bench2 = Benchmark("Transcript Feature Extraction")
    bench2.run(
        engineer.extract_transcript_features,
        transcripts,
        iterations=100)

    # Benchmark embedding extraction
    bench3 = Benchmark("Embedding Feature Extraction")
    bench3.run(engineer.extract_embedding_features, embeddings, iterations=100)

    # Benchmark full feature vector
    bench4 = Benchmark("Complete Feature Vector Build")
    bench4.run(
        engineer.build_feature_vector,
        keyword_hits,
        transcripts,
        embeddings,
        iterations=50)


if __name__ == "__main__":
    print("=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)

    benchmark_feature_extraction()

    print("\n" + "=" * 60)
    print("BENCHMARKS COMPLETE")
    print("=" * 60)
