#!/usr/bin/env python
"""
Fix all remaining code quality warnings
"""
import re
import os
import sys


def fix_f_strings(content):
    """Remove f prefix from strings without placeholders"""
    pattern = r'f(["\'])([^{}]*?)\1'

    def replace_func(match):
        quote = match.group(1)
        text = match.group(2)
        if '{' not in text and '}' not in text:
            return f'{quote}{text}{quote}'
        return match.group(0)

    return re.sub(pattern, replace_func, content)


def fix_comparison_to_false(content):
    """Fix is False to is False"""
    return re.sub(r'==\s*False\b', 'is False', content)


def remove_unused_variables(content):
    """Mark unused variables with _ prefix"""
    # This is complex, better let autoflake handle it
    return content


def fix_file(file_path):
    """Apply all fixes to a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original = content

        # Apply fixes
        content = fix_f_strings(content)
        content = fix_comparison_to_false(content)

        # Only write if changed
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed: {file_path}")
            return True

        return False

    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False


def main():
    """Fix all Python files"""
    fixed_count = 0

    for root, dirs, files in os.walk('.'):
        # Skip directories
        dirs[:] = [d for d in dirs if d not in {'.venv', '__pycache__', '.git', 'node_modules'}]

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_file(file_path):
                    fixed_count += 1

    print(f"\n✅ Fixed {fixed_count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
