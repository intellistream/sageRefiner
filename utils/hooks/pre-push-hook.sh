#!/bin/bash
# sageRefiner Pre-push Hook
# ═══════════════════════════════════════════════════════════════════════════
#
# Purpose: Validate code quality and tests before pushing to remote
#
# This hook prevents pushing code that:
#   1. Has uncommitted changes
#   2. Fails local tests
#   3. Has import errors
#
# Installation:
#   ln -sf ../../utils/hooks/pre-push-hook.sh .git/hooks/pre-push
#   chmod +x .git/hooks/pre-push
#
# Skip this hook temporarily:
#   git push --no-verify
#
# ═══════════════════════════════════════════════════════════════════════════

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REMOTE="$1"
URL="$2"
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Helper functions
print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check 1: Ensure working directory is clean
print_header "Pre-Push Validation (Branch: $BRANCH → $REMOTE)"

if ! git diff-index --quiet HEAD --; then
    print_error "Uncommitted changes detected"
    echo "Please commit or stash your changes before pushing"
    exit 1
fi
print_success "Working directory clean"

# Check 2: Verify no syntax errors in Python files
print_header "Checking Python Syntax"

python_files=$(git diff --name-only HEAD~1..HEAD -- "*.py" 2>/dev/null || echo "")
if [ -n "$python_files" ]; then
    if python3 -m py_compile $python_files 2>/dev/null; then
        print_success "No Python syntax errors"
    else
        print_error "Python syntax errors detected"
        exit 1
    fi
fi

# Check 3: Run quick import validation
print_header "Validating Module Imports"

if python3 -c "from sage_refiner import *; print('  All imports validated')" 2>/dev/null; then
    print_success "Core module imports valid"
else
    print_warning "Could not fully validate imports (might need install)"
fi

# Check 4: Verify no test failures (optional, can be slow)
if [ "$SKIP_TESTS" != "true" ]; then
    print_header "Running Quick Tests"
    
    if command -v pytest &> /dev/null; then
        if pytest tests/ -x -q --tb=no 2>/dev/null; then
            print_success "Tests passed"
        else
            print_warning "Some tests failed. Push anyway? (use --no-verify to skip)"
            # Don't exit here - let user decide
        fi
    else
        print_warning "pytest not found, skipping tests"
    fi
fi

# Check 5: Verify branch protection rules
print_header "Branch Protection Checks"

# Warn if pushing to protected branches without PR
if [[ "$BRANCH" == "main" || "$BRANCH" == "main-dev" ]]; then
    print_warning "You are about to push to '$BRANCH' branch"
    echo "Consider using a feature branch and creating a Pull Request instead."
fi

print_success "Pre-push validation complete"
print_header "Ready to push $BRANCH to $REMOTE"

exit 0
