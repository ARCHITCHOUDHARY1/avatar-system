

import os
import re
from pathlib import Path

# Unicode to ASCII mapping (comprehensive)
UNICODE_REPLACEMENTS = {
    # Checkmarks and status
    '[OK]': '[OK]',
    '[OK]': '[OK]',
    '[OK]': '[OK]',
    '[ERROR]': '[ERROR]',
    '[ERROR]': '[ERROR]',
    '[ERROR]': '[ERROR]',
    '[WARNING]': '[WARNING]',
    '[WARNING]': '[WARNING]',
    
    # Arrows
    '->': '->',
    '<-': '<-',
    '<->': '<->',
    '=>': '=>',
    '<=': '<=',
    '^': '^',
    'v': 'v',
    '^': '^',
    'v': 'v',
    
    # Stars and symbols
    '*': '*',
    '*': '*',
    '*': '*',
    '[NEW]': '[NEW]',
    '[INFO]': '[INFO]',
    '[TOOL]': '[TOOL]',
    '[NOTE]': '[NOTE]',
    '[STATS]': '[STATS]',
    '[TARGET]': '[TARGET]',
    '[LAUNCH]': '[LAUNCH]',
    '[BUG]': '[BUG]',
    '[SEARCH]': '[SEARCH]',
    '[TEST]': '[TEST]',
    '[VIDEO]': '[VIDEO]',
    '[AUDIO]': '[AUDIO]',
    '[AI]': '[AI]',
    '[CODE]': '[CODE]',
    '[PACKAGE]': '[PACKAGE]',
    '[SUCCESS]': '[SUCCESS]',
    '[FAST]': '[FAST]',
    
    # Math symbols
    '>=': '>=',
    '<=': '<=',
    '!=': '!=',
    '~=': '~=',
    '*': '*',
    '/': '/',
    '+/-': '+/-',
    
    # Quotes
    '"': '"',
    '"': '"',
    ''""": '"',
    '"': '"',
    
    # Misc
    '...': '...',
    '--': '--',
    '-': '-',
    '*': '*',
    '.': '.',
    ' deg': ' deg',
    '(c)': '(c)',
    '(R)': '(R)',
    '(TM)': '(TM)',
}

def remove_unicode_from_file(file_path):
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original = content
        
        # Apply replacements
        for unicode_char, ascii_replacement in UNICODE_REPLACEMENTS.items():
            content = content.replace(unicode_char, ascii_replacement)
        
        # Remove any remaining non-ASCII characters (except in strings/comments)
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Keep docstrings and comments mostly intact
            if '"""' in line or "'''" in line or line.strip().startswith('#'):
                # Only remove problematic unicode
                cleaned_line = line
                for char in line:
                    if ord(char) > 127 and char not in [' ', '\t', '\r', '\n']:
                        # Check if it's in our mapping
                        if char not in UNICODE_REPLACEMENTS:
                            cleaned_line = cleaned_line.replace(char, '?')
                cleaned_lines.append(cleaned_line)
            else:
                # For code lines, be more aggressive
                cleaned_line = ''.join(char if ord(char) < 128 else 
                                     UNICODE_REPLACEMENTS.get(char, '?') 
                                     for char in line)
                cleaned_lines.append(cleaned_line)
        
        content = '\n'.join(cleaned_lines)
        
        if content != original:
            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Cleaned"
        
        return False, "No changes"
        
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Clean all Python files in project"""
    project_root = Path(__file__).parent.parent
    
    print("="*70)
    print("COMPLETE UNICODE REMOVAL")
    print("="*70)
    
    # Find all Python files
    python_files = [f for f in project_root.rglob("*.py") 
                   if '__pycache__' not in str(f) and '.venv' not in str(f)]
    
    print(f"\nFound {len(python_files)} Python files")
    print("\nCleaning files...")
    
    cleaned_count = 0
    for file_path in python_files:
        changed, status = remove_unicode_from_file(file_path)
        if changed:
            cleaned_count += 1
            rel_path = file_path.relative_to(project_root)
            print(f"[CLEANED] {rel_path}")
    
    print("\n" + "="*70)
    print(f"COMPLETE: Cleaned {cleaned_count} files")
    print("="*70)
    
    # Verify
    print("\nVerifying... checking for remaining unicode...")
    remaining = []
    for file_path in python_files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            for char in content:
                if ord(char) > 127 and char not in ['\r', '\n', '\t']:
                    if file_path not in remaining:
                        remaining.append(file_path)
                    break
    
    if remaining:
        print(f"\n[WARNING] {len(remaining)} files still have unicode:")
        for f in remaining[:10]:  # Show first 10
            print(f"  - {f.relative_to(project_root)}")
    else:
        print("\n[OK] All files are clean!")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
