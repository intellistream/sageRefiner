#!/usr/bin/env bash
# Installation logic for SageRefiner.
# Supports multiple installation profiles via pyproject.toml extras.

# ============================================================================
# Installation Profile Selection
# ============================================================================
INSTALL_PROFILE=""

select_install_profile() {
    print_header "$(get_msg select_profile)"

    echo "  1) $(get_msg profile_basic)"
    echo "  2) $(get_msg profile_full)"
    echo "  3) $(get_msg profile_benchmark)"
    echo ""

    read -p "  $(get_msg enter_choice): " -n 1 -r profile_choice
    echo ""
    echo ""

    case $profile_choice in
        1) INSTALL_PROFILE="." ;;
        2) INSTALL_PROFILE=".[full]" ;;
        3) INSTALL_PROFILE=".[benchmark]" ;;
        *)
            if [[ "$LANG_SETTING" == "zh" ]]; then
                print_warning "无效选择，使用基础配置"
            else
                print_warning "Invalid choice, using basic profile"
            fi
            INSTALL_PROFILE="."
            ;;
    esac

    export INSTALL_PROFILE
    echo "[INSTALL] Selected profile: $INSTALL_PROFILE" >> "$LOG_FILE"
}

# ============================================================================
# Installation
# ============================================================================
install_sage_refiner() {
    print_header "$(get_msg installing)"

    local profile_name
    case $INSTALL_PROFILE in
        ".") profile_name="basic" ;;
        ".[full]") profile_name="full" ;;
        ".[benchmark]") profile_name="benchmark" ;;
        *) profile_name="basic" ;;
    esac

    if [[ "$LANG_SETTING" == "zh" ]]; then
        print_info "安装配置: ${CYAN}$profile_name${NC}"
        print_info "使用 pyproject.toml 中定义的依赖"
    else
        print_info "Profile: ${CYAN}$profile_name${NC}"
        print_info "Using dependencies from pyproject.toml"
    fi
    echo ""

    echo "=== Installation Started at $(date) ===" >> "$LOG_FILE"
    echo "Profile: $profile_name ($INSTALL_PROFILE)" >> "$LOG_FILE"

    # Run pip install in background with spinner
    $PYTHON_CMD -m pip install -e "$INSTALL_PROFILE" >> "$LOG_FILE" 2>&1 &
    local pip_pid=$!

    # Simple spinner
    local spin='-\|/'
    local i=0
    while kill -0 $pip_pid 2>/dev/null; do
        i=$(( (i+1) % 4 ))
        if [[ "$LANG_SETTING" == "zh" ]]; then
            printf "\r  ${spin:$i:1} 安装中...  "
        else
            printf "\r  ${spin:$i:1} Installing...  "
        fi
        sleep 0.3
    done

    # Wait for pip to finish and get exit code
    wait $pip_pid
    local exit_code=$?

    printf "\r"  # Clear spinner line
    echo ""     # New line

    if [ $exit_code -eq 0 ]; then
        print_success "$(get_msg install_success)"
        echo "=== Installation Completed at $(date) ===" >> "$LOG_FILE"
        return 0
    else
        print_error "$(get_msg install_failed): $LOG_FILE"
        echo "=== Installation FAILED at $(date) ===" >> "$LOG_FILE"
        return 1
    fi
}
