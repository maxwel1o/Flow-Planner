#!/bin/bash
# NPU环境构建和编译脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Flow Planner NPU Environment Setup"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. 检查系统环境
echo ""
print_info "Step 1: Checking system environment..."
echo ""

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Python version: $PYTHON_VERSION"

# 检查pip
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif python3 -m pip --version &> /dev/null; then
    print_info "pip3 not in PATH, using 'python3 -m pip' as fallback."
    PIP_CMD="python3 -m pip"
else
    print_error "pip3 not found. Please install pip first."
    exit 1
fi

# 2. 检查NPU环境
echo ""
print_info "Step 2: Checking NPU environment..."
echo ""

# 检查CANN环境变量
if [ -z "$ASCEND_TOOLKIT_HOME" ]; then
    print_warn "ASCEND_TOOLKIT_HOME not set. CANN may not be installed."
    print_info "Please install CANN Toolkit from: https://www.hiascend.com/software/aiengine"
else
    print_info "CANN Toolkit path: $ASCEND_TOOLKIT_HOME"
fi

# 检查NPU设备
if [ -d "/dev/davinci0" ]; then
    print_info "NPU device found: /dev/davinci0"
else
    print_warn "NPU device not found. Training will fall back to CPU/CUDA."
fi

# 3. 安装PyTorch和torch_npu
echo ""
print_info "Step 3: Installing PyTorch and torch_npu..."
echo ""

# 卸载可能冲突的包
print_info "Removing conflicting packages..."
$PIP_CMD uninstall -y torch torchvision torch_npu 2>/dev/null || true

# 安装PyTorch和torch_npu (版本必须对应)
TORCH_VERSION="2.3.0"
TORCH_NPU_VERSION="2.3.1"
TORCHVISION_VERSION="0.18.0"

print_info "Installing PyTorch ${TORCH_VERSION}..."
$PIP_CMD install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu

print_info "Installing torch_npu ${TORCH_NPU_VERSION}..."
if $PIP_CMD install torch-npu==${TORCH_NPU_VERSION} --no-deps; then
    print_info "torch_npu installed successfully!"
else
    print_warn "Failed to install torch_npu. NPU support may not be available."
    print_info "You can manually install torch_npu from: https://gitee.com/ascend/pytorch"
fi

# 4. 安装项目依赖
echo ""
print_info "Step 4: Installing project dependencies..."
echo ""

# 创建NPU版本的requirements
print_info "Creating NPU-compatible requirements..."
cat > requirements_npu.txt <<EOF
# Core dependencies
hydra-core==1.3.2
omegaconf==2.3.0
numpy
scipy==1.13.1
einops==0.8.0
timm==1.0.10

# Flow matching
flow-matching

# Training monitoring
tensorboard==2.11.2
wandb==0.17.4

# Optimization
casadi

# NPU dependencies
decorator
cloudpickle
ml-dtypes
tornado

# PyTorch (already installed)
# torch==2.3.0
# torchvision==0.18.0
EOF

# 安装依赖
print_info "Installing dependencies from requirements_npu.txt..."
$PIP_CMD install -r requirements_npu.txt

# 5. 验证安装
echo ""
print_info "Step 5: Verifying installation..."
echo ""

# 验证PyTorch
print_info "Checking PyTorch..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# 验证torch_npu
print_info "Checking torch_npu..."
python3 << 'EOF'
try:
    import torch
    import torch_npu
    print(f"torch_npu version: {torch_npu.__version__}")
    print(f"NPU available: {torch.npu.is_available()}")
    if torch.npu.is_available():
        print(f"NPU device count: {torch.npu.device_count()}")
        for i in range(torch.npu.device_count()):
            print(f"  NPU {i}: {torch.npu.get_device_name(i)}")
except ImportError:
    print("torch_npu not installed. NPU support unavailable.")
EOF

# 验证其他依赖
print_info "Checking other dependencies..."
python3 << 'EOF'
import sys
dependencies = [
    'hydra',
    'omegaconf',
    'numpy',
    'scipy',
    'einops',
    'timm',
    'tensorboard',
    'wandb',
    'flow_matching'
]

for dep in dependencies:
    try:
        __import__(dep)
        print(f"✓ {dep}")
    except ImportError as e:
        print(f"✗ {dep}: {e}")
EOF

# 6. 测试NPU兼容性
echo ""
print_info "Step 6: Testing NPU compatibility..."
echo ""

if [ -f "npu_compat.py" ]; then
    print_info "Running npu_compat.py test..."
    python3 npu_compat.py
else
    print_warn "npu_compat.py not found. Skipping compatibility test."
fi

# 7. 构建项目
echo ""
print_info "Step 7: Building project..."
echo ""

# 安装项目
if [ -f "setup.py" ]; then
    print_info "Installing Flow Planner..."
    $PIP_CMD install -e .
else
    print_warn "setup.py not found. Skipping project installation."
fi

# 8. 创建NPU配置文件
echo ""
print_info "Step 8: Creating NPU configuration..."
echo ""

# 创建NPU训练配置
if [ -d "flow_planner/script" ]; then
    print_info "Creating NPU training config..."
    cat > flow_planner/script/flow_planner_npu.yaml <<EOF
# NPU训练配置
# 继承标准配置并修改设备设置

defaults:
  - flow_planner_standard
  - _self_

# 设备设置
device: npu  # 使用NPU设备

# 分布式训练设置
ddp:
  distributed: true
  init_process_group: hccl  # 使用华为集合通信库

# 训练参数(可根据NPU内存调整)
train:
  batch_size: 1024  # NPU可能需要减小batch size
  epoch: 400
EOF
    print_info "NPU config created: flow_planner/script/flow_planner_npu.yaml"
fi

# 9. 完成
echo ""
echo "=========================================="
print_info "NPU environment setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Verify NPU is working: python3 npu_compat.py"
echo "  2. Run verification script: bash verify_npu.sh"
echo "  3. Start training with NPU: python -m flow_planner.trainer --config-name flow_planner_npu"
echo ""

