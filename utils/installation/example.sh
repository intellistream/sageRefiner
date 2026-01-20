#!/usr/bin/env bash
# User-facing example and documentation display for SageRefiner quickstart.

# ============================================================================
# Quick Example
# ============================================================================
show_quick_example_and_docs() {
    echo ""
    if [[ "$LANG_SETTING" == "zh" ]]; then
        print_header "快速示例"
        echo "  在 Python 中尝试:"
    else
        print_header "Quick Example"
        echo "  Try this in Python:"
    fi
    echo ""
    echo -e "  ${BLUE}from sage_refiner.algorithms.llmlingua2 import LLMLingua2Compressor${NC}"
    echo ""
    if [[ "$LANG_SETTING" == "zh" ]]; then
        echo -e "  ${BLUE}# 创建压缩器${NC}"
    else
        echo -e "  ${BLUE}# Create compressor${NC}"
    fi
    echo -e "  ${BLUE}compressor = LLMLingua2Compressor()${NC}"
    echo ""
    if [[ "$LANG_SETTING" == "zh" ]]; then
        echo -e "  ${BLUE}# 压缩文本${NC}"
    else
        echo -e "  ${BLUE}# Compress text${NC}"
    fi
    echo -e "  ${BLUE}result = compressor.compress(${NC}"
    echo -e "  ${BLUE}    context=\"Your long context text here...\",${NC}"
    echo -e "  ${BLUE}    question=\"What is the main topic?\"${NC}"
    echo -e "  ${BLUE})${NC}"
    echo -e "  ${BLUE}print(result.compressed_context)${NC}"
    echo ""

    if [[ "$LANG_SETTING" == "zh" ]]; then
        print_header "文档"
        print_info "README: ./README.md"
        print_info "示例: ./examples/"
        print_info "基准测试: ./benchmarks/"
    else
        print_header "Documentation"
        print_info "README: ./README.md"
        print_info "Examples: ./examples/"
        print_info "Benchmarks: ./benchmarks/"
    fi
    echo ""
}

# ============================================================================
# Run Example Script
# ============================================================================
maybe_run_example_script() {
    if [[ "$LANG_SETTING" == "zh" ]]; then
        read -p "  运行示例脚本? (y/N) " -n 1 -r
    else
        read -p "  Run example script? (y/N) " -n 1 -r
    fi
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [[ -f "examples/basic_compression.py" ]]; then
            if [[ "$LANG_SETTING" == "zh" ]]; then
                print_info "正在运行示例..."
            else
                print_info "Running example..."
            fi
            echo "=== Example Execution Started at $(date) ===" >> "$LOG_FILE"
            $PYTHON_CMD examples/basic_compression.py 2>&1 | tee -a "$LOG_FILE" | head -50
            echo "=== Example Execution Completed at $(date) ===" >> "$LOG_FILE"
        else
            if [[ "$LANG_SETTING" == "zh" ]]; then
                print_warning "示例脚本未找到"
            else
                print_warning "Example script not found"
            fi
        fi
    fi
}

# ============================================================================
# Final Summary
# ============================================================================
show_final_summary() {
    echo ""
    if [[ "$LANG_SETTING" == "zh" ]]; then
        print_header "安装完成！"
        print_success "SageRefiner 已准备就绪!"
        echo ""
        print_info "导入路径: ${BLUE}from sage_refiner import ...${NC}"
        print_info "包名: isage-refiner (v$VERSION)"
        print_info "安装日志保存到: $LOG_FILE"
        echo ""
        print_info "更多信息请访问: https://github.com/intellistream/SageRefiner"
    else
        print_header "Setup Complete!"
        print_success "SageRefiner is ready to use!"
        echo ""
        print_info "Import path: ${BLUE}from sage_refiner import ...${NC}"
        print_info "Package: isage-refiner (v$VERSION)"
        print_info "Installation log saved to: $LOG_FILE"
        echo ""
        print_info "For more information, visit: https://github.com/intellistream/SageRefiner"
    fi
    echo ""
    echo "=== Installation Script Completed at $(date) ===" >> "$LOG_FILE"
}
