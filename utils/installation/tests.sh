#!/usr/bin/env bash
# Basic test runner for SageRefiner quickstart.

# ============================================================================
# Test Runner
# ============================================================================
maybe_run_basic_tests() {
    echo ""
    if [[ "$LANG_SETTING" == "zh" ]]; then
        read -p "  运行基础测试? (y/N) " -n 1 -r
    else
        read -p "  Run basic tests? (y/N) " -n 1 -r
    fi
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Create test log directory
        mkdir -p .sage/test

        if [[ "$LANG_SETTING" == "zh" ]]; then
            print_info "正在运行测试..."
        else
            print_info "Running tests..."
        fi
        echo "=== Tests Started at $(date) ===" >> "$LOG_FILE"

        # Check if pytest is available
        if ! command -v pytest &>/dev/null; then
            if [[ "$LANG_SETTING" == "zh" ]]; then
                print_warning "pytest 未安装，正在安装..."
            else
                print_warning "pytest not installed, installing..."
            fi
            $PYTHON_CMD -m pip install pytest >> "$LOG_FILE" 2>&1
        fi

        # Run tests
        if pytest tests/ -v --tb=short 2>&1 | tee -a "$LOG_FILE" | tail -30; then
            if [[ "$LANG_SETTING" == "zh" ]]; then
                print_success "测试完成"
            else
                print_success "Tests completed"
            fi
        else
            if [[ "$LANG_SETTING" == "zh" ]]; then
                print_warning "部分测试可能失败 (检查是否安装了依赖)"
            else
                print_warning "Some tests may have failed (check if dependencies are installed)"
            fi
        fi
        echo "=== Tests Completed at $(date) ===" >> "$LOG_FILE"

        # Clean up benchmark artifacts
        rm -rf .benchmarks 2>/dev/null || true
    fi
}
