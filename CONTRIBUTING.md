# Contributing to Avatar System Orchestrator

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🤝 Ways to Contribute

- 🐛 Report bugs and issues
- 💡 Suggest new features or improvements
- 📝 Improve documentation
- 🔧 Submit pull requests with bug fixes or features
- ⭐ Star the repository

## 📋 Getting Started

### 1. Fork & Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/avatar-system.git
cd avatar-system
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models (see MODELS_SETUP.md)
python download_wav2lip_model.py
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

## 🔧 Development Guidelines

### Code Style

- Follow **PEP 8** Python style guide
- Use **type hints** for function arguments and returns
- Add **docstrings** to all classes and functions
- Keep functions focused and under 50 lines when possible

### Example:

```python
def process_audio(audio_path: str, *, sample_rate: int = 16000) -> np.ndarray:
    """
    Process audio file and extract features.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default: 16000)
        
    Returns:
        Audio features as numpy array
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
    """
    # Implementation
    pass
```

### Testing

```bash
# Run tests before submitting PR
python -m pytest tests/

# Test specific feature
python test_generation_api.py
```

### Commit Messages

Use conventional commit format:

```
feat: Add batch processing support
fix: Resolve memory leak in video generation  
docs: Update installation instructions
refactor: Simplify emotion detection logic
test: Add unit tests for audio processor
```

## 📬 Submitting Changes

### Pull Request Process

1. **Update Documentation**: If adding features, update README.md
2. **Add Tests**: Include tests for new functionality
3. **Check Code Quality**: Run linters and formatters
4. **Update CHANGELOG**: Note your changes
5. **Create PR**: Use the PR template

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tested locally
- [ ] Added/updated tests
- [ ] All tests passing

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🐛 Reporting Bugs

### Bug Report Format

```markdown
**Describe the bug**
Clear description of the issue

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. Use input files '...'
3. See error

**Expected behavior**
What should happen

**Environment**
- OS: Windows/Linux/Mac
- Python version: 3.9/3.10/3.11
- GPU: NVIDIA GTX/CPU only

**Logs**
```
Paste relevant error logs
```
```

## 💡 Feature Requests

Use GitHub Issues with:
- Clear title describing the feature
- **Why** the feature is needed
- **How** it would work
- Potential implementation details (optional)

## 📚 Documentation

Help improve docs:
- Fix typos and clarity issues
- Add examples and tutorials
- Expand API documentation
- Create video walkthroughs

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Eligible for "Contributor" badge

## 📞 Questions?

- 💬 GitHub Discussions
- 🐛 GitHub Issues
- 📧 Project maintainers

## 📜 License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

---

**Thank you for making Avatar System better!** 🎉
