#!/usr/bin/env python
"""
Fix unused global in embeddings-service
"""
import sys

def fix_embeddings_service():
    """Remove unused global embeddings_client"""
    file_path = 'services/embeddings-service/app/main.py'
    
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Remove line 113 (or any line with unused global)
        new_lines = []
        for i, line in enumerate(lines, 1):
            # Skip line if it's ONLY "global embeddings_client"
            if line.strip() == 'global embeddings_client':
                print(f"Removed line {i}: {line.strip()}")
                continue
            new_lines.append(line)
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"✅ Fixed {file_path}")
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = fix_embeddings_service()
    sys.exit(0 if success else 1)
