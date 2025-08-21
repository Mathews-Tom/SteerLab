# Contributing to SteerLab

We welcome contributions from the community! Whether you're fixing a bug, adding a new feature, or improving documentation, your help is appreciated.

## Development Environment Setup

To get started, set up your environment using `uv`, our recommended package manager.

```bash
# 1. Fork and clone the repository
git clone https://github.com/Mathews-Tom/steerlab.git
cd steerlab

# 2. Create and activate a virtual environment
uv venv
source .venv/bin/activate

# 3. Install editable package with dev dependencies
uv pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install
```

## Code Quality and Standards

We use a suite of modern Python tools to ensure high-quality, consistent code. The `pre-commit` hooks will automatically run these tools on every commit.

- **Formatting**: `ruff format`
- **Linting**: `ruff check`
- **Type Checking**: `mypy`

You can run all checks manually at any time:

```bash
pre-commit run --all-files
```

## Running Tests

All contributions must pass the existing test suite. If you are adding a new feature, please include tests for it.

```bash
# Run all tests
pytest

# Run tests for a specific module
pytest steerlab/core/

# Run tests with coverage
pytest --cov=steerlab
```

## Submitting Changes

1. **Create a Feature Branch**:

    ```bash
    git checkout -b feature/my-new-feature
    ```

2. **Make Your Changes**: Write your code and add corresponding tests.
3. **Ensure Checks Pass**: Make sure `pre-commit run --all-files` and `pytest` complete successfully.
4. **Commit Your Changes**: Write a clear and descriptive commit message.
5. **Push to Your Fork**:

    ```bash
    git push origin feature/my-new-feature
    ```

6. **Submit a Pull Request**: Open a pull request from your fork to the `main` branch of the original repository. Provide a clear description of the changes you've made.

## Documentation

For changes that affect the public API or user-facing features, please update the documentation in the `docs/` directory.

Thank you for contributing to SteerLab!
