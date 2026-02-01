# Pre-commit Hooks & Development Guide

## Overview

sageRefiner uses **pre-commit hooks** to maintain code quality, consistency, and security before
code is committed and pushed to the repository.

This document describes the available hooks and how to use them.

______________________________________________________________________

## What are Pre-commit Hooks?

Pre-commit hooks are automated checks that run **before you create a git commit**. They help to:

- Enforce consistent code style across the project
- Catch common errors early (syntax, imports, etc.)
- Prevent accidental commits of sensitive data
- Ensure code quality standards are met
- Reduce back-and-forth reviews

Pre-push hooks additionally validate code before pushing to remote repository.

______________________________________________________________________

## Installation

### Option 1: Automatic Installation (Recommended)

During the initial project setup, run:

```bash
cd /path/to/sageRefiner
utils/installation/install.sh
```

The installer will prompt you to install pre-commit hooks. Choose **yes** to proceed.

### Option 2: Manual Installation

1. **Install pre-commit framework:**

```bash
pip install pre-commit
```

2. **Install commit hooks:**

```bash
pre-commit install
```

3. **Install push hooks (optional but recommended):**

```bash
pre-commit install --hook-type pre-push
```

Or manually:

```bash
ln -sf ../../utils/hooks/pre-push-hook.sh .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

______________________________________________________________________

## Available Hooks

### Pre-commit Hooks (run on `git commit`)

The `.pre-commit-config.yaml` file defines all pre-commit hooks:

#### 1. **General File Checks**

- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml**: Validate YAML syntax
- **check-json**: Validate JSON syntax
- **check-toml**: Validate TOML syntax
- **check-added-large-files**: Prevent committing large files (>1MB)
- **check-merge-conflict**: Detect merge conflict markers
- **check-case-conflict**: Detect case conflicts in filenames
- **mixed-line-ending**: Normalize line endings (LF)
- **detect-private-key**: Detect private keys

#### 2. **Python Code Quality (Ruff)**

- **ruff check**: Fast Python linting (replaces flake8, isort, etc.)
- **ruff format**: Code formatting (replaces black)

Both with auto-fix enabled.

#### 3. **YAML Formatting**

- **pretty-format-yaml**: Auto-format YAML files
- Excludes `.github/` directory to avoid modifying workflows

#### 4. **Markdown Formatting**

- **mdformat**: Format markdown with GFM extensions
- Wraps lines at 100 characters
- Excludes CHANGELOG.md

#### 5. **Security**

- **detect-secrets**: Scan for API keys, tokens, and secrets
- Excludes template files and fixtures

______________________________________________________________________

### Pre-push Hooks (run on `git push`)

Custom hook script: `utils/hooks/pre-push-hook.sh`

Validates:

- ✓ No uncommitted changes
- ✓ No Python syntax errors
- ✓ Module imports are valid
- ⚠️ Test suite passes (warning only, doesn't block)
- ⚠️ No direct pushes to protected branches (warning)

______________________________________________________________________

## Usage

### Running Hooks Manually

#### Run all hooks on changed files:

```bash
pre-commit run
```

#### Run all hooks on all files:

```bash
pre-commit run --all-files
```

#### Run specific hook:

```bash
pre-commit run ruff --all-files
```

#### Update hook versions:

```bash
pre-commit autoupdate
```

### Skipping Hooks

#### Skip pre-commit hooks for a single commit:

```bash
git commit --no-verify
```

#### Skip pre-push hooks:

```bash
git push --no-verify
```

⚠️ **Use with caution** - skipping hooks defeats their purpose!

______________________________________________________________________

## Configuration Details

### Ruff Configuration

Ruff rules are configured in `pyproject.toml` under `[tool.ruff]`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
# Enabled rules...
```

To customize which rules to enforce:

```bash
# Check current configuration
ruff rule --all

# Run with specific rules
ruff check --select=E,W,F --ignore=E501
```

### YAML Configuration

YAML files are formatted with:

- 2-space indentation
- Preserved quotes (single/double)
- Auto-fixed indentation

### Markdown Configuration

Markdown files are formatted with:

- 100-character line wrap
- GFM (GitHub Flavored Markdown) extensions
- Preserves code blocks

______________________________________________________________________

## Troubleshooting

### Hook Installation Issues

**Problem:** `command not found: pre-commit`

**Solution:**

```bash
pip install pre-commit
```

**Problem:** `No such file or directory: .git/hooks/pre-commit`

**Solution:**

```bash
pre-commit install
```

### Hook Failures

#### Ruff formatting conflicts:

If ruff-check and ruff-format conflict:

```bash
# Ruff format first, then check
ruff format .
pre-commit run ruff-format
pre-commit run ruff
```

#### Trailing whitespace in markdown:

Pre-commit might add markdown line breaks. Re-stage:

```bash
git add .
git commit
```

#### Large file warnings:

If you have legitimate large files, exclude them:

```bash
git add --force large-file.bin
git commit
```

### Pre-commit Cache Issues

Clear cache if hooks aren't updating:

```bash
pre-commit clean
pre-commit run --all-files
```

______________________________________________________________________

## CI/CD Integration

The GitHub Actions workflow (`.github/workflows/ci-tests.yml`) automatically runs pre-commit hooks
on:

- **Pull Requests**: Only changed files are checked
- **Push**: All files are checked

This ensures code quality standards even if developers skip local hooks.

### Version Consistency Check

The CI also verifies that local `ruff` version matches the version in `.pre-commit-config.yaml`:

```bash
# Update versions if mismatch occurs
pre-commit autoupdate
```

______________________________________________________________________

## Best Practices

### For Contributors

1. ✅ Always install pre-commit hooks in your local environment
1. ✅ Run `pre-commit run --all-files` before submitting PRs
1. ✅ Commit the auto-formatted code (ruff will format it)
1. ✅ Don't skip hooks unless absolutely necessary
1. ✅ Keep `.pre-commit-config.yaml` in sync with actual tool versions

### For Maintainers

1. ✅ Update hook versions regularly: `pre-commit autoupdate`
1. ✅ Review hook failures in CI and address root causes
1. ✅ Ensure `.pre-commit-config.yaml` is reviewed in PRs
1. ✅ Document any new local hooks

______________________________________________________________________

## Quick Reference

| Command                                   | Purpose                    |
| ----------------------------------------- | -------------------------- |
| `pre-commit install`                      | Install pre-commit hooks   |
| `pre-commit install --hook-type pre-push` | Install push hooks         |
| `pre-commit run --all-files`              | Run all hooks on all files |
| `pre-commit run <hook-id>`                | Run specific hook          |
| `pre-commit autoupdate`                   | Update hook versions       |
| `pre-commit clean`                        | Clear hook cache           |
| `git commit --no-verify`                  | Skip pre-commit hooks      |
| `git push --no-verify`                    | Skip pre-push hooks        |

______________________________________________________________________

## Related Files

- **Configuration**: `.pre-commit-config.yaml`
- **Code Style**: `pyproject.toml` ([tool.ruff] section)
- **CI Configuration**: `.github/workflows/ci-tests.yml`
- **Push Hook Script**: `utils/hooks/pre-push-hook.sh`
- **Installation Script**: `utils/installation/precommit.sh`

______________________________________________________________________

## Additional Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [SAGE Project Standards](https://github.com/intellistream/sageLLM)

______________________________________________________________________

## Support

For issues with pre-commit hooks:

1. Check this guide's troubleshooting section
1. Run `pre-commit clean && pre-commit run --all-files`
1. Check GitHub Issues for similar problems
1. Reach out to the development team
