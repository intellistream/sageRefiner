#!/bin/bash
# sageRefiner Pre-commit Quick Setup
# ═════════════════════════════════════════════════════════════════════════════
#
# This script quickly sets up pre-commit and pre-push hooks for sageRefiner
#
# Usage:
#   bash utils/hooks/setup-hooks.sh
#   or
#   ./utils/hooks/setup-hooks.sh
#
# ═════════════════════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Main script
print_header "sageRefiner Pre-commit Hooks Setup"

# Check if we're in the right directory
if [ ! -f ".pre-commit-config.yaml" ]; then
    print_error "Not in sageRefiner root directory"
    echo "Please run this script from the sageRefiner root directory"
    exit 1
fi

# Step 1: Check Python
print_header "Step 1: Checking Python"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 not found"
    exit 1
fi

# Step 2: Install pre-commit if needed
print_header "Step 2: Installing pre-commit framework"
if command -v pre-commit &> /dev/null; then
    PRE_COMMIT_VERSION=$(pre-commit --version)
    print_success "pre-commit already installed: $PRE_COMMIT_VERSION"
else
    print_info "Installing pre-commit..."
    pip install pre-commit
    if [ $? -eq 0 ]; then
        print_success "pre-commit installed"
    else
        print_error "Failed to install pre-commit"
        exit 1
    fi
fi

# Step 3: Install pre-commit hooks
print_header "Step 3: Installing pre-commit hooks"
if pre-commit install; then
    print_success "Pre-commit hooks installed"
else
    print_error "Failed to install pre-commit hooks"
    exit 1
fi

# Step 4: Install pre-push hooks
print_header "Step 4: Installing pre-push hooks"
if [ -f "utils/hooks/pre-push-hook.sh" ]; then
    if [ -L ".git/hooks/pre-push" ] || [ -f ".git/hooks/pre-push" ]; then
        print_info "Pre-push hook already exists"
    else
        if ln -sf ../../utils/hooks/pre-push-hook.sh .git/hooks/pre-push; then
            chmod +x .git/hooks/pre-push
            print_success "Pre-push hook installed"
        else
            print_error "Failed to install pre-push hook"
            exit 1
        fi
    fi
else
    print_error "Pre-push hook script not found"
    exit 1
fi

# Step 5: Verify installation
print_header "Step 5: Verifying installation"

if [ -f ".git/hooks/pre-commit" ]; then
    print_success ".git/hooks/pre-commit exists"
else
    print_error ".git/hooks/pre-commit not found"
fi

if [ -L ".git/hooks/pre-push" ] || [ -f ".git/hooks/pre-push" ]; then
    print_success ".git/hooks/pre-push exists"
else
    print_error ".git/hooks/pre-push not found"
fi

# Step 6: Quick test (optional)
print_header "Step 6: Testing hooks"
if pre-commit run --all-files --dry-run > /dev/null 2>&1; then
    print_success "Pre-commit hooks are functional"
else
    print_info "Hooks need dependencies (will be auto-installed on first commit)"
fi

# Final message
print_header "✓ Setup Complete!"
echo -e "${GREEN}Pre-commit and pre-push hooks are now installed!${NC}\n"

echo -e "${BLUE}Quick Reference:${NC}"
echo "  • Run all hooks:          ${YELLOW}pre-commit run --all-files${NC}"
echo "  • Run specific hook:      ${YELLOW}pre-commit run <hook-id>${NC}"
echo "  • Update hook versions:   ${YELLOW}pre-commit autoupdate${NC}"
echo "  • Skip commit hooks:      ${YELLOW}git commit --no-verify${NC}"
echo "  • Skip push hooks:        ${YELLOW}git push --no-verify${NC}"
echo ""
echo -e "${BLUE}For more information:${NC}"
echo "  • Read PRE_COMMIT_GUIDE.md"
echo "  • Check .pre-commit-config.yaml"
echo ""

exit 0
