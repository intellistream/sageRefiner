#!/usr/bin/env bash
# Verification helpers for SageRefiner installation.

# ============================================================================
# Verification
# ============================================================================
verify_sage_refiner_install() {
    print_header "$(get_msg verifying)"

    echo "=== Verification Started at $(date) ===" >> "$LOG_FILE"

    local verify_output
    verify_output=$($PYTHON_CMD -c "from sage_refiner import __version__; print(f'sage_refiner import OK, version={__version__}')" 2>&1)
    echo "$verify_output" >> "$LOG_FILE"

    if echo "$verify_output" | grep -q "sage_refiner import OK"; then
        print_success "$(get_msg verify_passed)"
        echo "=== Verification PASSED at $(date) ===" >> "$LOG_FILE"
        return 0
    else
        print_error "$(get_msg verify_failed)"
        echo "=== Verification FAILED at $(date) ===" >> "$LOG_FILE"

        if [[ "$LANG_SETTING" == "zh" ]]; then
            print_info "尝试诊断问题..."
        else
            print_info "Trying to diagnose the issue..."
        fi
        $PYTHON_CMD -c "from sage_refiner import __version__" 2>&1 | tee -a "$LOG_FILE" || true
        echo ""
        return 1
    fi
}

# ============================================================================
# Version Display
# ============================================================================
VERSION=""

show_installed_version() {
    VERSION=$($PYTHON_CMD -c "from sage_refiner import __version__; print(__version__)" 2>/dev/null || echo "unknown")
    export VERSION

    print_info "$(get_msg installed_version): ${CYAN}$VERSION${NC}"
    echo "Installed version: $VERSION" >> "$LOG_FILE"
}
