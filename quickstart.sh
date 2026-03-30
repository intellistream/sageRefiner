#!/usr/bin/env bash
# quickstart.sh — sageRefiner dev environment setup
#
# Usage:
#   ./quickstart.sh               # dev mode (default): hooks + .[dev]  (includes [full])
#   ./quickstart.sh --full        # ML backends only: .[full]  (torch, transformers)
#   ./quickstart.sh --standard    # core deps only: no extras
#   ./quickstart.sh --yes         # non-interactive (assume yes)
#   ./quickstart.sh --doctor      # diagnose environment issues
#
# Install matrix:
#   (default / --dev)  pip install -e .[dev]   ← includes [full] via self-ref
#   --full             pip install -e .[full]
#   --standard         pip install -e .
#
# Rules:
#   - NEVER creates a new venv. Must be called in an existing non-venv environment.
#   - Installs hooks via direct copy from hooks/.

set -e

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ─── Arguments ────────────────────────────────────────────────────────────────
EXTRAS="[dev]"   # default — dev includes [full] via pyproject self-reference
DOCTOR=false
YES=false
for arg in "$@"; do
    case "$arg" in
        --doctor)   DOCTOR=true ;;
        --standard) EXTRAS="" ;;
        --full)     EXTRAS="[full]" ;;
        --dev)      EXTRAS="[dev]" ;;
        --yes|-y)   YES=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${BLUE}  sageRefiner — Quick Start${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ─── Doctor ───────────────────────────────────────────────────────────────────
if [ "$DOCTOR" = true ]; then
    echo -e "${BOLD}${BLUE}Environment Diagnosis${NC}"
    echo ""
    echo -e "${YELLOW}Python:${NC} $(python3 --version 2>/dev/null || echo 'NOT FOUND')"
    echo -e "${YELLOW}Conda env:${NC} ${CONDA_DEFAULT_ENV:-none}"
    echo -e "${YELLOW}Venv:${NC} ${VIRTUAL_ENV:-none}"
    echo -e "${YELLOW}ruff:${NC} $(ruff --version 2>/dev/null || echo 'NOT FOUND')"
    echo -e "${YELLOW}pytest:${NC} $(pytest --version 2>/dev/null || echo 'NOT FOUND')"
    echo ""
    echo -e "${YELLOW}Git hooks installed:${NC}"
    for h in pre-commit pre-push post-commit; do
        if [ -f "$PROJECT_ROOT/.git/hooks/$h" ]; then
            echo -e "  ${GREEN}✓ $h${NC}"
        else
            echo -e "  ${RED}✗ $h${NC}"
        fi
    done
    exit 0
fi

# ─── Step 0: Require an active non-venv environment ──────────────────────────
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${RED}  ❌ Detected Python venv: $VIRTUAL_ENV${NC}"
    echo -e "${YELLOW}  → This repository forbids venv/.venv usage.${NC}"
    echo -e "${YELLOW}  → Activate an existing conda environment instead.${NC}"
    echo ""
    exit 1
fi

if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${RED}  ❌ No Python environment detected.${NC}"
    echo -e "${YELLOW}  → Activate an existing conda environment first:${NC}"
    echo -e "     conda create -n sage python=3.11 && conda activate sage"
    echo ""
    echo -e "${RED}  ⚠️  NEVER run this script without an active environment.${NC}"
    echo -e "${RED}  ⚠️  This script will NOT create a new environment for you.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✅ Environment: ${CONDA_DEFAULT_ENV}${NC}"
echo ""

# ─── Step 1: Python version check ────────────────────────────────────────────
echo -e "${YELLOW}${BOLD}Step 1/3: Python version${NC}"
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" || {
    echo -e "${RED}✗ Python $PYTHON_VERSION is too old (requires >= 3.10)${NC}"
    exit 1
}
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
echo ""

# ─── Step 2: Install Git Hooks ───────────────────────────────────────────────
echo -e "${YELLOW}${BOLD}Step 2/3: Installing Git Hooks${NC}"

HOOKS_DIR="$PROJECT_ROOT/.git/hooks"
TEMPLATE_HOOKS="$PROJECT_ROOT/hooks"

if [ ! -d "$HOOKS_DIR" ]; then
    echo -e "${YELLOW}⚠  .git directory not found — skipping hooks (not a git repo?)${NC}"
else
    if [ -d "$TEMPLATE_HOOKS" ]; then
        for hook in pre-commit pre-push post-commit; do
            if [ -f "$TEMPLATE_HOOKS/$hook" ]; then
                cp "$TEMPLATE_HOOKS/$hook" "$HOOKS_DIR/$hook"
                chmod +x "$HOOKS_DIR/$hook"
                echo -e "${GREEN}✓ $hook installed${NC}"
            fi
        done
    else
        echo -e "${YELLOW}⚠  hooks/ directory not found — skipping${NC}"
    fi
fi
echo ""

# ─── Step 3: Install package ─────────────────────────────────────────────────
echo -e "${YELLOW}${BOLD}Step 3/3: Installing package (editable)${NC}"
if [ -n "$EXTRAS" ]; then
    echo -e "  ${CYAN}pip install -e .${EXTRAS}${NC}"
    pip install -e ".${EXTRAS}" --quiet 2>/dev/null || pip install -e . --quiet
else
    echo -e "  ${CYAN}pip install -e .${NC}  (minimal)"
    pip install -e . --quiet
fi
echo -e "${GREEN}✓ Package installed in editable mode${EXTRAS:+ with extras: $EXTRAS}${NC}"
echo ""

echo -e "${GREEN}${BOLD}✓ Setup complete!${NC}"
echo ""
echo -e "${BLUE}${BOLD}Next steps:${NC}"
echo -e "  ${CYAN}pytest tests/${NC}                    — run tests (requires [full] extras)"
echo -e "  ${CYAN}ruff check src/${NC}                  — lint"
echo -e "  ${CYAN}./quickstart.sh --full${NC}            — reinstall with torch + transformers"
echo -e "  ${CYAN}./quickstart.sh --standard${NC}        — install core deps only (no extras)"
echo -e "  ${CYAN}./quickstart.sh --doctor${NC}          — diagnose environment"
echo ""
