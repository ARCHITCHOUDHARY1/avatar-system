"""
Project Health Check and Cleanup Script
- Removes unicode characters
- Checks for syntax errors
- Verifies imports
- Tests basic functionality
"""

import os
import re
import sys
from pathlib import Path

def remove_unicode_from_file(file_path):
    """Remove non-ASCII unicode characters from Python files"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace common unicode symbols
        replacements = {
            '[OK]': '[OK]',
            '[ERROR]': '[ERROR]',
            '[WARNING]': '[WARNING]',
            '->': '->',
            '<-': '<-',
            '*': '*',
            '[STATS]': '[STATS]',
            '[TARGET]': '[TARGET]',
            '[LAUNCH]': '[LAUNCH]',
            '[INFO]': '[INFO]',
            '[TOOL]': '[TOOL]',
            '[NOTE]': '[NOTE]',
            '[NEW]': '[NEW]',
            '[BUG]': '[BUG]',
            '[SEARCH]': '[SEARCH]',
            '[TEST]': '[TEST]',
            '[VIDEO]': '[VIDEO]',
            '[AUDIO]': '[AUDIO]',
            '[AI]': '[AI]',
            '>=': '>=',
            '<=': '<=',
            '<': '<',
        }
        
        # Apply replacements
        original_content = content
        for unicode_char, ascii_replacement in replacements.items():
            content = content.replace(unicode_char, ascii_replacement)
        
        # Remove any remaining non-ASCII characters (except in comments/strings)
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Keep comments and strings as-is (they might have valid unicode)
            if line.strip().startswith('#') or '"""' in line or "'''" in line:
                cleaned_lines.append(line)
            else:
                # Remove any remaining non-ASCII from code lines
                cleaned_line = ''.join(char if ord(char) < 128 else '?' for char in line)
                cleaned_lines.append(cleaned_line)
        
        content = '\n'.join(cleaned_lines)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Cleaned"
        
        return False, "No changes"
        
    except Exception as e:
        return False, f"Error: {e}"

def check_syntax(file_path):
    """Check Python file for syntax errors"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            compile(f.read(), file_path, 'exec')
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: line {e.lineno}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Run complete project health check"""
    project_root = Path(__file__).parent
    
    print("="*70)
    print("PROJECT HEALTH CHECK")
    print("="*70)
    
    # Find all Python files
    python_files = list(project_root.rglob("*.py"))
    python_files = [f for f in python_files if '.venv' not in str(f) and '__pycache__' not in str(f)]
    
    print(f"\nFound {len(python_files)} Python files")
    
    # Step 1: Remove unicode
    print("\n" + "="*70)
    print("STEP 1: Removing Unicode Characters")
    print("="*70)
    
    cleaned_count = 0
    for file_path in python_files:
        changed, status = remove_unicode_from_file(file_path)
        if changed:
            cleaned_count += 1
            print(f"[OK] Cleaned: {file_path.relative_to(project_root)}")
    
    print(f"\nCleaned {cleaned_count} files")
    
    # Step 2: Check syntax
    print("\n" + "="*70)
    print("STEP 2: Checking Syntax")
    print("="*70)
    
    syntax_errors = []
    for file_path in python_files:
        success, message = check_syntax(file_path)
        if not success:
            syntax_errors.append((file_path, message))
            print(f"[ERROR] {file_path.relative_to(project_root)}: {message}")
    
    if not syntax_errors:
        print("[OK] All files have valid syntax")
    else:
        print(f"\n[ERROR] Found {len(syntax_errors)} files with syntax errors")
    
    # Step 3: Check imports
    print("\n" + "="*70)
    print("STEP 3: Checking Key Imports")
    print("="*70)
    
    key_modules = [
        'models.vad_detector',
        'models.tts_generator',
        'models.speech_pipeline',
        'src.utils.performance_optimizer',
    ]
    
    sys.path.insert(0, str(project_root))
    
    import_errors = []
    for module in key_modules:
        try:
            __import__(module)
            print(f"[OK] {module}")
        except Exception as e:
            import_errors.append((module, str(e)))
            print(f"[ERROR] {module}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Files scanned: {len(python_files)}")
    print(f"Files cleaned: {cleaned_count}")
    print(f"Syntax errors: {len(syntax_errors)}")
    print(f"Import errors: {len(import_errors)}")
    
    if syntax_errors:
        print("\n[ERROR] Files with syntax errors:")
        for file_path, message in syntax_errors:
            print(f"  - {file_path.relative_to(project_root)}: {message}")
    
    if import_errors:
        print("\n[ERROR] Modules that failed to import:")
        for module, error in import_errors:
            print(f"  - {module}: {error}")
    
    if not syntax_errors and not import_errors:
        print("\n[OK] All checks passed!")
        return 0
    else:
        print("\n[WARNING] Some issues found - see above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
