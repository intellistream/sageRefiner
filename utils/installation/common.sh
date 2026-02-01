#!/usr/bin/env bash
# Common utilities for SageRefiner installation scripts.
# Includes: colors, logging, messages, and environment checks.

# ============================================================================
# ANSI Colors
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

# ============================================================================
# Bilingual Messages
# ============================================================================
declare -A MSG_EN
declare -A MSG_ZH

# General messages
MSG_EN[welcome]="Welcome to SageRefiner Setup"
MSG_ZH[welcome]="欢迎使用 SageRefiner 安装脚本"

MSG_EN[select_lang]="Select language / 选择语言: 1) English  2) 中文"
MSG_ZH[select_lang]="Select language / 选择语言: 1) English  2) 中文"

MSG_EN[invalid_choice]="Invalid choice, using English"
MSG_ZH[invalid_choice]="无效选择，使用英文"

# Environment check messages
MSG_EN[checking_env]="Checking Environment"
MSG_ZH[checking_env]="检查环境"

MSG_EN[not_in_conda]="Not running in a conda environment!"
MSG_ZH[not_in_conda]="当前不在 conda 环境中！"

MSG_EN[activate_conda]="Please activate your conda environment first:"
MSG_ZH[activate_conda]="请先激活您的 conda 环境："

MSG_EN[in_conda]="Running in conda environment"
MSG_ZH[in_conda]="当前在 conda 环境中"

MSG_EN[checking_python]="Checking Python version..."
MSG_ZH[checking_python]="检查 Python 版本..."

MSG_EN[python_version_ok]="Python version OK"
MSG_ZH[python_version_ok]="Python 版本正确"

MSG_EN[python_version_fail]="Python 3.11+ required, found"
MSG_ZH[python_version_fail]="需要 Python 3.11+，当前版本"

MSG_EN[checking_pip]="Checking pip path..."
MSG_ZH[checking_pip]="检查 pip 路径..."

MSG_EN[pip_path_ok]="pip is from conda environment"
MSG_ZH[pip_path_ok]="pip 来自 conda 环境"

MSG_EN[pip_path_fail]="pip is NOT from conda environment!"
MSG_ZH[pip_path_fail]="pip 不是来自 conda 环境！"

MSG_EN[pip_fix_hint]="Expected pip from conda, but found"
MSG_ZH[pip_fix_hint]="预期 pip 来自 conda，但发现"

MSG_EN[pip_fix_suggest]="Try removing ~/.local/bin/pip if exists, then restart terminal"
MSG_ZH[pip_fix_suggest]="尝试删除 ~/.local/bin/pip（如果存在），然后重启终端"

MSG_EN[env_check_passed]="Environment check passed"
MSG_ZH[env_check_passed]="环境检查通过"

# Installation messages
MSG_EN[select_profile]="Select Installation Profile"
MSG_ZH[select_profile]="选择安装配置"

MSG_EN[profile_basic]="Basic - Core package only"
MSG_ZH[profile_basic]="基础 - 仅核心包"

MSG_EN[profile_full]="Full - With all optional dependencies"
MSG_ZH[profile_full]="完整 - 包含所有可选依赖"

MSG_EN[profile_benchmark]="Benchmark - For running benchmarks"
MSG_ZH[profile_benchmark]="基准测试 - 用于运行基准测试"

MSG_EN[enter_choice]="Enter choice [1-3]"
MSG_ZH[enter_choice]="输入选择 [1-3]"

MSG_EN[installing]="Installing SageRefiner"
MSG_ZH[installing]="正在安装 SageRefiner"

MSG_EN[install_success]="Installation successful"
MSG_ZH[install_success]="安装成功"

MSG_EN[install_failed]="Installation failed. Check log for details"
MSG_ZH[install_failed]="安装失败。查看日志了解详情"

# Verification messages
MSG_EN[verifying]="Verifying Installation"
MSG_ZH[verifying]="验证安装"

MSG_EN[verify_passed]="Import verification passed"
MSG_ZH[verify_passed]="导入验证通过"

MSG_EN[verify_failed]="Import verification failed"
MSG_ZH[verify_failed]="导入验证失败"

MSG_EN[installed_version]="Installed version"
MSG_ZH[installed_version]="已安装版本"

# Summary messages
MSG_EN[setup_complete]="Setup Complete!"
MSG_ZH[setup_complete]="安装完成！"

MSG_EN[ready_to_use]="SageRefiner is ready to use!"
MSG_ZH[ready_to_use]="SageRefiner 已准备就绪！"

MSG_EN[import_path]="Import path"
MSG_ZH[import_path]="导入路径"

MSG_EN[log_saved]="Installation log saved to"
MSG_ZH[log_saved]="安装日志保存到"

# ============================================================================
# Language Selection
# ============================================================================
LANG_SETTING="en"

select_language() {
    echo ""
    echo "${MSG_EN[select_lang]}"
    read -p "> " -n 1 -r lang_choice
    echo ""
    case $lang_choice in
        1) LANG_SETTING="en" ;;
        2) LANG_SETTING="zh" ;;
        *)
            echo "${MSG_EN[invalid_choice]}"
            LANG_SETTING="en"
            ;;
    esac
    export LANG_SETTING
}

get_msg() {
    local key=$1
    if [[ "$LANG_SETTING" == "zh" ]]; then
        echo "${MSG_ZH[$key]}"
    else
        echo "${MSG_EN[$key]}"
    fi
}

# ============================================================================
# Logging
# ============================================================================
LOG_DIR=""
LOG_FILE=""

init_logging() {
    LOG_DIR=".sage/installation/sage_refiner"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/install_$(date +%Y%m%d_%H%M%S).log"
    export LOG_DIR LOG_FILE

    echo "=== SageRefiner Installation Log ===" >> "$LOG_FILE"
    echo "Started at: $(date)" >> "$LOG_FILE"
    echo "Conda env: ${CONDA_DEFAULT_ENV:-none}" >> "$LOG_FILE"
    echo "Python: $(python --version 2>&1)" >> "$LOG_FILE"
    echo "========================================" >> "$LOG_FILE"
}

# ============================================================================
# Print Helpers
# ============================================================================
print_info() {
    echo -e "  ${BLUE}→${NC} $1"
}

print_success() {
    echo -e "  ${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "  ${RED}✗${NC} $1"
}

print_header() {
    local msg=$1
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    printf "║  %-68s  ║\n" "$msg"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
}

# ============================================================================
# Environment Checks
# ============================================================================
PYTHON_CMD=""

check_conda_environment() {
    print_header "$(get_msg checking_env)"

    # Check if in conda environment
    if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" == "base" ]]; then
        print_error "$(get_msg not_in_conda)"
        echo ""
        print_info "$(get_msg activate_conda)"
        echo -e "    ${CYAN}conda activate <your-env-name>${NC}"
        echo ""
        exit 1
    fi
    print_success "$(get_msg in_conda): ${CYAN}$CONDA_DEFAULT_ENV${NC}"
    echo "[ENV] Conda environment: $CONDA_DEFAULT_ENV" >> "$LOG_FILE"

    # Check Python version
    print_info "$(get_msg checking_python)"
    local python_version
    python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    local major minor
    major=$(echo "$python_version" | cut -d. -f1)
    minor=$(echo "$python_version" | cut -d. -f2)

    if [[ "$major" -lt 3 ]] || [[ "$major" -eq 3 && "$minor" -lt 11 ]]; then
        print_error "$(get_msg python_version_fail): $python_version"
        echo "[ENV] Python version check FAILED: $python_version" >> "$LOG_FILE"
        exit 1
    fi
    print_success "$(get_msg python_version_ok): ${CYAN}$python_version${NC}"
    echo "[ENV] Python version: $python_version" >> "$LOG_FILE"

    # Check pip path
    print_info "$(get_msg checking_pip)"
    local pip_path
    pip_path=$(which pip 2>/dev/null || echo "not found")
    local conda_prefix="${CONDA_PREFIX:-}"

    if [[ -z "$conda_prefix" ]] || [[ ! "$pip_path" == "$conda_prefix"* ]]; then
        print_error "$(get_msg pip_path_fail)"
        print_info "$(get_msg pip_fix_hint): ${RED}$pip_path${NC}"
        print_info "$(get_msg pip_fix_suggest)"
        echo "[ENV] pip path check FAILED: $pip_path" >> "$LOG_FILE"
        exit 1
    fi
    print_success "$(get_msg pip_path_ok): ${CYAN}$pip_path${NC}"
    echo "[ENV] pip path: $pip_path" >> "$LOG_FILE"

    # Set Python command
    PYTHON_CMD="python"
    export PYTHON_CMD

    echo ""
    print_success "$(get_msg env_check_passed)"
    echo "[ENV] Environment check PASSED" >> "$LOG_FILE"
}
