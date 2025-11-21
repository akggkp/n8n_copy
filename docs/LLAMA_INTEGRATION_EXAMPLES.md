# Llama Integration Examples

This script demonstrates how to interact with the new Llama-powered endpoints to get trading strategy examples and generate strategies.

## Python Script for Strategy Generation

```python
import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8003")
KEYWORDS = ["RSI", "MACD", "Bollinger Bands", "Support", "Resistance", "breakout"]

def get_examples(keyword, top_k=5):
    """1. Get examples for a specific keyword"""
    print(f"\n--- 1. Fetching examples for '{keyword}' ---")
    try:
        response = requests.get(
            f"{API_BASE}/llama/examples",
            params={"keyword": keyword, "top_k": top_k, "include_embeddings": False},
            timeout=30
        )
        response.raise_for_status()
        examples = response.json()
        print(f"✓ Found {examples.get('total', 0)} examples for '{keyword}'")
        return examples.get('examples', [])
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching examples for {keyword}: {e}")
        return []

def generate_strategy(keyword, top_k=5, temperature=0.7):
    """2. Generate a trading strategy using Llama"""
    print(f"\n--- 2. Generating strategy for '{keyword}' ---")
    try:
        response = requests.post(
            f"{API_BASE}/llama/generate-strategy",
            params={"keyword": keyword, "top_k": top_k, "temperature": temperature},
            timeout=180
        )
        response.raise_for_status()
        strategy = response.json()
        print(f"✓ Strategy generated for '{keyword}'")
        print(f"\n{'='*60}\nTRADING STRATEGY: {keyword.upper()}\n{'='*60}")
        print(strategy.get('strategy', 'No strategy content returned.'))
        print(f"{'='*60}\n")
        return strategy
    except requests.exceptions.RequestException as e:
        print(f"✗ Error generating strategy for {keyword}: {e}")
        return None

def get_summary(keyword, top_k=3):
    """3. Get a concept summary for a keyword"""
    print(f"\n--- 3. Getting summary for '{keyword}' ---")
    try:
        response = requests.get(
            f"{API_BASE}/llama/summarize/{keyword}",
            params={"top_k": top_k},
            timeout=90
        )
        response.raise_for_status()
        summary = response.json()
        print(f"✓ Summary generated for '{keyword}'")
        print(f"\nCONCEPT SUMMARY: {keyword.upper()}")
        print(summary.get('summary', 'No summary content returned.'))
        return summary
    except requests.exceptions.RequestException as e:
        print(f"✗ Error getting summary for {keyword}: {e}")
        return None

def process_keyword_in_parallel(keyword):
    """Wrapper function for parallel execution"""
    print(f"\n===== Processing Keyword: {keyword.upper()} =====")
    get_examples(keyword)
    generate_strategy(keyword)
    get_summary(keyword)
    print(f"===== Finished Processing: {keyword.upper()} =====")

if __name__ == "__main__":
    # --- Direct Ollama Integration Example ---
    print("\n\n--- Direct Ollama Integration Example ---")
    
    # 1. Get examples from API
    direct_examples = get_examples("MACD", top_k=5)
    
    if direct_examples:
        # 2. Build a custom prompt
        context = "\n".join([
            f"- {ex.get('transcript', '')[:150]}... (at {ex.get('timestamp', 0)}s)"
            for ex in direct_examples
        ])
        
        prompt = f"""Based on these trading video excerpts about MACD:
        {context}
        
        Generate a step-by-step trading strategy using MACD. Include:
        1. Entry signals (e.g., MACD line crosses above signal line)
        2. Exit signals (e.g., MACD line crosses below signal line)
        3. Stop loss placement (e.g., below recent swing low)
        4. Position sizing (e.g., risk 1% of account)
        """
        
        # 3. Send prompt directly to Ollama
        try:
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            print(f"\n--- Sending custom prompt to Ollama at {ollama_url} ---")
            
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": "llama2", "prompt": prompt, "stream": False},
                timeout=120
            )
            response.raise_for_status()
            strategy = response.json().get("response", "No response from Ollama.")
            
            print("\n--- Custom Strategy from Direct Ollama Call ---")
            print(strategy)
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error communicating directly with Ollama: {e}")

    # --- Batch Processing Multiple Keywords in Parallel ---
    print("\n\n--- Batch Processing Multiple Keywords in Parallel ---")
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(process_keyword_in_parallel, KEYWORDS)

    print("\n\nAll tasks completed.")
```