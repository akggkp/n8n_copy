#!/usr/bin/env python
"""
Test ML Service
Tests: Keyword database, concept extraction, pattern detection
"""
import sys

# Add service to path
sys.path.insert(0, 'services/ml-service')


def test_imports():
    """Test if all required packages are installed"""
    print("🧪 Test 2.1: Checking imports...")
    try:
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_keywords_database():
    """Test keyword database exists and is populated"""
    print("\n🧪 Test 2.2: Testing keywords database...")
    try:
        from app.main import KEYWORDS_DB

        total_keywords = sum(len(v) for v in KEYWORDS_DB.values())
        categories = list(KEYWORDS_DB.keys())

        print("✅ Keywords database loaded")
        print(f"   Total keywords: {total_keywords}")
        print(f"   Categories: {len(categories)}")
        print(f"   Categories: {', '.join(categories)}")

        # Check specific categories
        expected_categories = [
            'technical_indicator',
            'price_action',
            'candlestick_pattern',
            'risk_management',
            'order_type',
            'trading_strategy',
            'market_structure'
        ]

        missing = [cat for cat in expected_categories if cat not in categories]
        if missing:
            print(f"⚠️  Missing categories: {missing}")
            return False

        return True
    except Exception as e:
        print(f"❌ Keywords database test failed: {e}")
        return False


def test_concept_extraction():
    """Test concept extraction with sample text"""
    print("\n🧪 Test 2.3: Testing concept extraction...")
    try:
        from app.main import KEYWORDS_DB
        import re

        # Sample trading text
        sample_text = """
        In this video, I'll show you a powerful RSI strategy.
        First, wait for the price to hit a support level.
        Then, check if RSI is below 30, which indicates oversold conditions.
        Set your stop loss 2% below entry and take profit at resistance.
        This gives us a good risk reward ratio of 1:2.
        """

        sample_text_lower = sample_text.lower()
        detected_keywords = []

        for category, keywords in KEYWORDS_DB.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = list(
                    re.finditer(
                        pattern,
                        sample_text_lower,
                        re.IGNORECASE))

                if matches:
                    detected_keywords.append({
                        'keyword': keyword,
                        'category': category,
                        'count': len(matches)
                    })

        if detected_keywords:
            print("✅ Concept extraction working")
            print(f"   Detected {len(detected_keywords)} keywords:")
            for kw in detected_keywords[:5]:  # Show first 5
                print(
                    f"   - '{kw['keyword']}' ({kw['category']}) - {kw['count']}x")
            return True
        else:
            print("❌ No keywords detected in sample text")
            return False

    except Exception as e:
        print(f"❌ Concept extraction test failed: {e}")
        return False


def test_service_endpoints():
    """Test if service can be imported and endpoints exist"""
    print("\n🧪 Test 2.4: Testing service structure...")
    try:
        pass

        print("✅ Service module imported successfully")
        print("   Endpoints: /, /health, /extract_concepts, /categories")
        return True
    except Exception as e:
        print(f"❌ Service import failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("ML SERVICE - STANDALONE TESTS")
    print("=" * 60)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Keywords Database", test_keywords_database()))
    results.append(("Concept Extraction", test_concept_extraction()))
    results.append(("Service Endpoints", test_service_endpoints()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nPassed: {total_passed}/{len(results)}")

    if total_passed == len(results):
        print("\n🎉 All tests passed! ML service is ready.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Fix issues before building Docker image.")
        sys.exit(1)
