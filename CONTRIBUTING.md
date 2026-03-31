# Contributing to ms1isotopes

Thank you for your interest in contributing to ms1isotopes!

## How to contribute

### Reporting bugs

Open an issue on [GitHub](https://github.com/VilenneFrederique/ms1isotopes/issues)
with a minimal reproducible example, your Python version, and package versions.

### Suggesting features

We welcome feature requests, particularly:
- Support for additional PTMs
- New output formats
- Integration with other proteomics tools
- DIA data support

Open an issue describing the feature and its use case.

### Code contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Run the test suite: `pytest tests/`
5. Submit a pull request

### Adding PTM support

To add a new modification, add its delta atomic composition to
`MODIFICATIONS` in `isotopes.py` and update the `parse_modifications()`
function in `extraction.py`. Include a test case.

## Development setup

```bash
git clone https://github.com/VilenneFrederique/ms1isotopes.git
cd ms1isotopes
pip install -e ".[dev]"
pytest tests/
```

## Code style

- Type hints on all public functions
- Docstrings in NumPy style
- No lines longer than 88 characters (black formatter)

## License

By contributing, you agree that your contributions will be licensed
under the GPLv3 license.
