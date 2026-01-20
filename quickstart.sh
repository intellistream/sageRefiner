#!/usr/bin/env bash
# SageRefiner Quick Start Script
# Modular installation following neuromem structure
#
# Usage: ./quickstart.sh
#
# Requirements:
#   - Conda environment activated (not base)
#   - Python 3.11+
#   - pip from conda environment (not ~/.local/bin/pip)
#
# Installation profiles (all from pyproject.toml):
#   1) Basic     - Core package only (pip install -e .)
#   2) Full      - All optional dependencies (pip install -e .[full])
#   3) Benchmark - For running benchmarks (pip install -e .[benchmark])

set -e

# ============================================================================
# Resolve script directory and source modules
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="$SCRIPT_DIR/utils/installation"

# Source all utility modules
source "$UTILS_DIR/common.sh"
source "$UTILS_DIR/install.sh"
source "$UTILS_DIR/verify.sh"
source "$UTILS_DIR/dependencies.sh"
source "$UTILS_DIR/precommit.sh"
source "$UTILS_DIR/tests.sh"
source "$UTILS_DIR/example.sh"

# ============================================================================
# Main
# ============================================================================
main() {
    clear

    # Welcome banner
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║                 ███████╗ █████╗  ██████╗ ███████╗                    ║"
    echo "║                 ██╔════╝██╔══██╗██╔════╝ ██╔════╝                    ║"
    echo "║                 ███████╗███████║██║  ███╗█████╗                      ║"
    echo "║                 ╚════██║██╔══██║██║   ██║██╔══╝                      ║"
    echo "║                 ███████║██║  ██║╚██████╔╝███████╗                    ║"
    echo "║                 ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝                    ║"
    echo "║                                                                      ║"
    echo "║                      R E F I N E R                                   ║"
    echo "║                                                                      ║"
    echo "║           Context Compression Algorithms for LLM Systems             ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""

    # Step 0: Language selection
    select_language

    # Step 1: Initialize logging
    init_logging

    # Step 2: Environment check (conda + pip path)
    check_conda_environment

    # Step 3: Select installation profile
    select_install_profile

    # Step 4: Install
    if ! install_sage_refiner; then
        exit 1
    fi

    # Step 5: Verify
    verify_sage_refiner_install
    show_installed_version

    # Step 6: Check optional dependencies
    check_optional_dependencies

    # Step 7: Optional - pre-commit hooks
    maybe_install_precommit

    # Step 8: Optional - run tests
    maybe_run_basic_tests

    # Step 9: Show example and documentation
    show_quick_example_and_docs
    maybe_run_example_script

    # Step 10: Final summary
    show_final_summary
}

# Run main function
main "$@"
