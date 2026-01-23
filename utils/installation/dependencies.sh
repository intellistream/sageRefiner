#!/usr/bin/env bash
# Optional dependency checks for SageRefiner.

# ============================================================================
# Dependency Checking
# ============================================================================
check_dependency() {
    local module=$1
    local description=$2
    local description_zh=$3

    if $PYTHON_CMD -c "import $module" 2>/dev/null; then
        if [[ "$LANG_SETTING" == "zh" ]]; then
            print_success "$module 已安装"
        else
            print_success "$module installed"
        fi
        echo "[DEPENDENCY] $module - installed" >> "$LOG_FILE"
    else
        if [[ "$LANG_SETTING" == "zh" ]]; then
            print_warning "$module 未安装 (可选: ${description_zh:-$description})"
        else
            print_warning "$module not installed (optional: $description)"
        fi
        echo "[DEPENDENCY] $module - NOT installed ($description)" >> "$LOG_FILE"
    fi
}

check_optional_dependencies() {
    echo ""
    if [[ "$LANG_SETTING" == "zh" ]]; then
        print_header "可选依赖检查"
    else
        print_header "Optional Dependencies Check"
    fi

    echo "=== Dependency Check Started at $(date) ===" >> "$LOG_FILE"

    # Core compression algorithm dependencies
    check_dependency "torch" "for neural compression models" "用于神经压缩模型"
    check_dependency "transformers" "for transformer models" "用于 Transformer 模型"
    check_dependency "llmlingua" "for LLMLingua2 compression" "用于 LLMLingua2 压缩"

    # Benchmark dependencies
    check_dependency "scipy" "for statistical analysis" "用于统计分析"
    check_dependency "matplotlib" "for visualization" "用于可视化"
    check_dependency "pandas" "for data analysis" "用于数据分析"
    check_dependency "datasets" "for HuggingFace datasets" "用于 HuggingFace 数据集"

    # SAGE ecosystem
    check_dependency "sage" "for SAGE CLI integration" "用于 SAGE CLI 集成"

    echo "=== Dependency Check Completed at $(date) ===" >> "$LOG_FILE"

    echo ""
    if [[ "$LANG_SETTING" == "zh" ]]; then
        print_info "所有必需的依赖都已通过 pyproject.toml 安装"
    else
        print_info "All required dependencies are installed via pyproject.toml"
    fi
    echo ""
}
