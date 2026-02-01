# Release & Publishing Guide

This guide explains how to release and publish new versions of sageRefiner to PyPI.

## Automated Publishing Process

sageRefiner uses **GitHub Actions** to automatically publish to PyPI when you push a version tag.

### Quick Release Workflow

1. **Update version** in `pyproject.toml`:

   ```toml
   version = "0.2.0"  # Update to new version
   ```

1. **Commit and push** the version change:

   ```bash
   git add pyproject.toml
   git commit -m "chore: Bump version to 0.2.0"
   git push origin main
   ```

1. **Create a git tag** matching the version:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

1. **GitHub Actions automatically**:

   - ✅ Verifies version matches the tag
   - ✅ Runs all tests and linting
   - ✅ Builds the distribution package
   - ✅ Publishes to PyPI
   - ✅ Creates a GitHub release

### Manual Publishing (Local)

If needed, you can publish locally:

```bash
# Build the package
python -m pip install build twine
python -m build

# Verify package
twine check dist/*

# Upload to TestPyPI (optional, for testing)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

## Version Numbering

sageRefiner follows **semantic versioning** (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Examples: `0.1.0`, `0.2.1`, `1.0.0`

## Pre-Release Checklist

Before releasing:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`
- [ ] Version updated in `pyproject.toml`
- [ ] Changelog updated (if maintained)
- [ ] Commit and push to main branch
- [ ] Tag matches version (v0.2.0 → version 0.2.0)

## CI/CD Publishing Pipeline

The `.github/workflows/publish.yml` workflow:

1. **Triggers** on tag push matching `v*.*.*`
1. **Verifies** tag version matches pyproject.toml
1. **Builds** distribution packages
1. **Publishes** to PyPI using OIDC authentication
1. **Creates** GitHub release with install instructions
1. **Notifies** with PyPI link

### Authentication

- Uses **GitHub OIDC** for secure, token-free PyPI authentication
- No credentials stored in repository
- Requires PyPI project to be configured for GitHub OIDC trust

## Troubleshooting

### Version Mismatch Error

```
Error: Version mismatch! Tag (v0.2.0) != pyproject.toml (0.1.0)
```

**Fix**: Ensure version in `pyproject.toml` matches the git tag:

```bash
# Check current version
grep '^version = ' pyproject.toml

# Update if needed and push again
git tag v0.2.0
```

### Failed Publishing

Check the GitHub Actions logs:

1. Go to: https://github.com/intellistream/sageRefiner/actions
1. Click the failed workflow
1. Check the "Publish to PyPI" step for details

### Manual Fix

If the automated workflow fails:

1. Fix the issue locally
1. Commit and push to main
1. Delete and recreate the tag:
   ```bash
   git tag -d v0.2.0
   git push origin :refs/tags/v0.2.0
   git tag v0.2.0
   git push origin v0.2.0
   ```

## PyPI Project Configuration

The package is published as **isage-refiner** to PyPI:

- **PyPI Project**: https://pypi.org/project/isage-refiner/
- **Installation**: `pip install isage-refiner`
- **Homepage**: https://github.com/intellistream/sageRefiner

## Release History

Published versions are available at:

- GitHub Releases: https://github.com/intellistream/sageRefiner/releases
- PyPI Releases: https://pypi.org/project/isage-refiner/#history
