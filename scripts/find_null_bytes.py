#!/usr/bin/env python
"""
Find Python files containing null bytes
"""
import os
import sys

def find_null_bytes(directory='.'):
    """Find all Python files with null bytes"""
    problematic_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip common directories
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.pytest_cache'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        if b'\x00' in content:
                            null_count = content.count(b'\x00')
                            problematic_files.append((file_path, null_count))
                            print(f"❌ FOUND: {file_path}")
                            print(f"   Null bytes: {null_count}")
                except Exception as e:
                    print(f"⚠️  Error reading {file_path}: {e}")
    
    return problematic_files

if __name__ == "__main__":
    print("=" * 60)
    print("Searching for Python files with null bytes...")
    print("=" * 60)
    
    problematic = find_null_bytes()
    
    if problematic:
        print(f"\n❌ Found {len(problematic)} problematic file(s):")
        for file_path, null_count in problematic:
            print(f"   - {file_path} ({null_count} null bytes)")
        print("\n💡 Fix: Delete these files and recreate them")
        sys.exit(1)
    else:
        print("\n✅ No files with null bytes found")
        sys.exit(0)
