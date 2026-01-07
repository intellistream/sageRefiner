#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project info
PROJECT_NAME="sageRefiner"
MIN_PYTHON_VERSION="3.11"

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   sageRefiner Quick Start                      ║"
echo "║     Intelligent Context Compression Library for LLM Systems    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to compare version numbers
version_ge() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

# Check Python version
echo -e "${YELLOW}[1/6] Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

if ! version_ge "$PYTHON_VERSION" "$MIN_PYTHON_VERSION"; then
    echo -e "${RED}Error: Python $MIN_PYTHON_VERSION or higher is required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version check passed${NC}\n"

# Check if in project directory
if [ ! -f "pyproject.toml" ] || [ ! -f "setup.py" ]; then
    echo -e "${RED}Error: Please run this script from the sageRefiner project root directory${NC}"
    exit 1
fi

# Check for existing virtual environment
echo -e "${YELLOW}[2/6] Virtual environment setup${NC}"

# Detect if already in a virtual environment
IN_VENV=false
VENV_TYPE=""

if [ -n "$CONDA_DEFAULT_ENV" ]; then
    IN_VENV=true
    VENV_TYPE="conda"
    echo -e "${GREEN}✓ Detected active conda environment: ${CONDA_DEFAULT_ENV}${NC}"
elif [ -n "$VIRTUAL_ENV" ]; then
    IN_VENV=true
    VENV_TYPE="virtualenv"
    echo -e "${GREEN}✓ Detected active virtual environment: ${VIRTUAL_ENV}${NC}"
fi

if [ "$IN_VENV" = true ]; then
    read -p "Create a new conda environment? [y/N]: " create_new
    create_new=${create_new:-N}
    
    if [[ $create_new =~ ^[Yy]$ ]]; then
        # Proceed to create new conda environment
        IN_VENV=false
    else
        echo "Using existing ${VENV_TYPE} environment for installation"
        echo -e "${GREEN}✓ Virtual environment check passed${NC}\n"
    fi
fi

if [ "$IN_VENV" = false ]; then
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
        echo "Please install Miniconda or Anaconda first"
        echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    # Ask about conda environment creation
    read -p "Create a new conda environment? (recommended) [Y/n]: " create_venv
    create_venv=${create_venv:-Y}

    if [[ $create_venv =~ ^[Yy]$ ]]; then
        read -p "Enter conda environment name (default: sagerefiner): " conda_env_name
        conda_env_name=${conda_env_name:-sagerefiner}

        # Check if conda environment already exists
        if conda env list | grep -q "^${conda_env_name} "; then
            echo "Conda environment '$conda_env_name' already exists"
            read -p "Remove and recreate? [y/N]: " recreate
            if [[ $recreate =~ ^[Yy]$ ]]; then
                echo "Removing existing conda environment..."
                conda env remove -n "$conda_env_name" -y
                echo "Creating new conda environment '$conda_env_name' with Python ${MIN_PYTHON_VERSION}..."
                conda create -n "$conda_env_name" python="${MIN_PYTHON_VERSION}" -y
            else
                echo "Using existing conda environment"
            fi
        else
            echo "Creating conda environment '$conda_env_name' with Python ${MIN_PYTHON_VERSION}..."
            conda create -n "$conda_env_name" python="${MIN_PYTHON_VERSION}" -y
        fi

        echo "Activating conda environment..."
        eval "$(conda shell.bash hook)"
        conda activate "$conda_env_name"
        echo -e "${GREEN}✓ Conda environment activated${NC}\n"
    else
        echo "Skipping conda environment creation"
        echo -e "${YELLOW}Warning: Installing without conda environment (not recommended)${NC}\n"
    fi
fi

# Upgrade pip
echo -e "${YELLOW}[3/6] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ pip upgraded${NC}\n"

# Select installation profile
echo -e "${YELLOW}[4/6] Select installation profile${NC}"
echo "Available profiles:"
echo "  1) Basic        - Core dependencies only"
echo "  2) Full         - All optional dependencies (vLLM + reranker)"
echo "  3) Development  - Full + development tools + benchmark dependencies"

read -p "Choose profile [1-3] (default: 1): " profile
profile=${profile:-1}

case $profile in
    1)
        INSTALL_CMD="pip install -e ."
        PROFILE_NAME="Basic"
        ;;
    2)
        INSTALL_CMD="pip install -e .[full]"
        PROFILE_NAME="Full"
        ;;
    3)
        INSTALL_CMD="pip install -e .[full,dev,benchmark]"
        PROFILE_NAME="Development"
        ;;
    *)
        echo -e "${RED}Invalid profile selection${NC}"
        exit 1
        ;;
esac

echo "Selected profile: $PROFILE_NAME"
echo -e "${GREEN}✓ Profile selected${NC}\n"

# Install the package
echo -e "${YELLOW}[5/6] Installing $PROJECT_NAME ($PROFILE_NAME profile)...${NC}"
echo "Running: $INSTALL_CMD"
echo "This may take several minutes..."
echo ""

if $INSTALL_CMD; then
    echo -e "${GREEN}✓ Installation completed successfully${NC}\n"
    
    # Install SAGE packages in stages if benchmark profile is selected
    if [[ $profile -eq 3 ]]; then
        echo -e "${BLUE}"
        echo "╔════════════════════════════════════════════════════════════════╗"
        echo "║          Installing SAGE Packages (Staged Installation)       ║"
        echo "╚════════════════════════════════════════════════════════════════╝"
        echo -e "${NC}\n"
        
        echo -e "${YELLOW}步骤 1/4: 安装 L1 基础包 (isage-common[embedding])${NC}"
        pip install --upgrade "isage-common[embedding]"
        echo -e "${GREEN}✓ L1 基础包安装完成${NC}\n"
        
        echo -e "${YELLOW}步骤 2/4: 安装 L2-L3 平台&核心包 (isage-platform, isage-kernel, isage-libs)${NC}"
        pip install --upgrade isage-platform isage-kernel isage-libs
        echo -e "${GREEN}✓ L2-L3 平台&核心包安装完成${NC}\n"
        
        echo -e "${YELLOW}步骤 3/4: 安装 L4 中间件包 (isage-middleware)${NC}"
        pip install --upgrade isage-middleware
        echo -e "${GREEN}✓ L4 中间件包安装完成${NC}\n"
        
        echo -e "${YELLOW}步骤 4/4: 安装 L5-L6 上层包 (isage-vdb, isage-data)${NC}"
        pip install --upgrade isage-vdb isage-data
        echo -e "${GREEN}✓ L5-L6 上层包安装完成${NC}\n"
        
        echo -e "${GREEN}✓ 所有 SAGE 包安装完成${NC}\n"
    fi
else
    echo -e "${RED}✗ Installation failed${NC}"
    exit 1
fi

# Run quick verification
echo -e "${YELLOW}[6/6] Verifying installation...${NC}"
if python3 -c "import sage_refiner; print(f'sageRefiner version: {sage_refiner.__version__}')"; then
    echo -e "${GREEN}✓ Installation verified${NC}\n"
else
    echo -e "${RED}✗ Verification failed${NC}"
    exit 1
fi

# Success message
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                 Installation Completed! 🎉                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo "Next steps:"
echo ""

# Show appropriate activation command only if not already in venv
if [ "$IN_VENV" = false ] && [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "1. Activate the conda environment:"
    echo -e "   ${BLUE}conda activate ${conda_env_name}${NC}"
    echo ""
    echo "2. Try a basic example:"
else
    echo "1. Try a basic example:"
fi
echo -e "   ${BLUE}python examples/basic_compression.py${NC}"
echo ""

if [ "$IN_VENV" = false ] && [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "3. Compare algorithms:"
else
    echo "2. Compare algorithms:"
fi
echo -e "   ${BLUE}python examples/algorithm_comparison.py${NC}"
echo ""

if [ "$IN_VENV" = false ] && [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "4. Run tests:"
else
    echo "3. Run tests:"
fi
echo -e "   ${BLUE}pytest tests/${NC}"
echo ""
echo "For more information, see README.md"
echo ""

# Optional: check GPU availability if Full or Development profile was selected
if [[ $profile -eq 2 || $profile -eq 3 ]]; then
    echo -e "${YELLOW}GPU Check:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo -e "${YELLOW}Warning: nvidia-smi not found. vLLM requires CUDA-capable GPU.${NC}"
    fi
    echo ""
fi

echo -e "${GREEN}Happy compressing! 🚀${NC}"
