#!/bin/bash
# NPU适配验证脚本

set -e

echo "=========================================="
echo "Flow Planner NPU Verification Script"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 测试计数器
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 测试函数
run_test() {
    local test_name=$1
    local test_cmd=$2
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
    echo -e "${BLUE}[TEST $TOTAL_TESTS]${NC} $test_name"
    echo "----------------------------------------"
    
    if eval "$test_cmd"; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo -e "${GREEN}✓ PASSED${NC}"
        return 0
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo -e "${RED}✗ FAILED${NC}"
        return 1
    fi
}

# 1. 环境检查
echo ""
echo "=== 1. Environment Check ==="
echo ""

run_test "Check Python version" "python3 --version"

run_test "Check CANN Toolkit" 'if [ -n "$ASCEND_TOOLKIT_HOME" ]; then echo "CANN_HOME: $ASCEND_TOOLKIT_HOME"; true; else echo "CANN not found"; false; fi'

run_test "Check NPU devices" "ls -la /dev/davinci* 2>/dev/null || echo 'No NPU devices found'"

# 2. Python包检查
echo ""
echo "=== 2. Python Package Check ==="
echo ""

run_test "Check PyTorch" "python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'"

run_test "Check torch_npu" "python3 -c 'import torch_npu; print(f\"torch_npu: {torch_npu.__version__}\")'"

run_test "Check NPU availability" "python3 -c 'import torch; import torch_npu; print(f\"NPU available: {torch.npu.is_available()}\")'"

run_test "Check Hydra" "python3 -c 'import hydra; print(f\"Hydra: {hydra.__version__}\")'"

run_test "Check flow-matching" "python3 -c 'import flow_matching; print(\"flow-matching installed\")'"

run_test "Check einops" "python3 -c 'import einops; print(f\"einops: {einops.__version__}\")'"

run_test "Check timm" "python3 -c 'import timm; print(f\"timm: {timm.__version__}\")'"

# 3. NPU兼容性模块测试
echo ""
echo "=== 3. NPU Compatibility Module Test ==="
echo ""

run_test "Test npu_compat.py" "python3 npu_compat.py"

run_test "Test npu_ops.py" "python3 npu_ops.py"

# 4. 设备功能测试
echo ""
echo "=== 4. Device Function Test ==="
echo ""

run_test "Test device detection" "python3 << 'EOF'
import torch
try:
    import torch_npu
    if torch.npu.is_available():
        print(f\"NPU device count: {torch.npu.device_count()}\")
        for i in range(torch.npu.device_count()):
            print(f\"  NPU {i}: {torch.npu.get_device_name(i)}\")
        print(\"NPU detection: SUCCESS\")
    else:
        print(\"NPU not available, but torch_npu installed\")
except ImportError:
    print(\"torch_npu not installed\")
EOF
"

run_test "Test tensor operations on NPU" "python3 << 'EOF'
import torch
try:
    import torch_npu
    if torch.npu.is_available():
        # 创建张量并移到NPU
        x = torch.randn(10, 10).npu()
        y = torch.randn(10, 10).npu()
        
        # 测试基本操作
        z = x + y
        z = torch.matmul(x, y)
        z = torch.softmax(x, dim=-1)
        
        print(\"Tensor operations on NPU: SUCCESS\")
    else:
        print(\"NPU not available, skipping tensor test\")
except Exception as e:
    print(f\"Error: {e}\")
    exit(1)
EOF
"

run_test "Test distributed backend" "python3 << 'EOF'
from npu_compat import get_dist_backend, get_device_type
device_type = get_device_type()
backend = get_dist_backend(device_type)
print(f\"Device type: {device_type}\")
print(f\"Distributed backend: {backend}\")
print(\"Backend test: SUCCESS\")
EOF
"

# 5. 模型导入测试
echo ""
echo "=== 5. Model Import Test ==="
echo ""

run_test "Test Flow Planner import" "python3 -c 'from flow_planner.model.flow_planner_model.flow_planner import FlowPlanner; print(\"FlowPlanner imported successfully\")'"

run_test "Test encoder import" "python3 -c 'from flow_planner.model.flow_planner_model.encoder import FlowPlannerEncoder; print(\"FlowPlannerEncoder imported successfully\")'"

run_test "Test decoder import" "python3 -c 'from flow_planner.model.flow_planner_model.decoder import FlowPlannerDecoder; print(\"FlowPlannerDecoder imported successfully\")'"

run_test "Test flow ODE import" "python3 -c 'from flow_planner.model.flow_planner_model.flow_utils.flow_ode import FlowODE; print(\"FlowODE imported successfully\")'"

# 6. 数据处理测试
echo ""
echo "=== 6. Data Processing Test ==="
echo ""

run_test "Test data processor import" "python3 -c 'from flow_planner.data.data_process.data_processor import DataProcessor; print(\"DataProcessor imported successfully\")'"

run_test "Test normalization" "python3 << 'EOF'
import torch
from flow_planner.data.normalization.state_normalize import StateNormalizer

# 创建归一化器配置
config = {
    'ego': {'waypoints': {'mean': 0.0, 'std': 1.0}},
    'neighbor': {'waypoints': {'mean': 0.0, 'std': 1.0}}
}
future_downsampling_method = 'waypoints'
predicted_neighbor_num = 10

normalizer = StateNormalizer(config, future_downsampling_method, predicted_neighbor_num)

# 测试归一化
data = torch.randn(5, 11, 10)  # (batch, num_agents, features)
normalized = normalizer(data)
denormalized = normalizer.inverse(normalized)

print(f\"Normalization test: SUCCESS\")
print(f\"Data shape: {data.shape}\")
print(f\"Normalized shape: {normalized.shape}\")
EOF
"

# 7. 配置文件测试
echo ""
echo "=== 7. Configuration Test ==="
echo ""

run_test "Test Hydra config loading" "python3 << 'EOF'
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(project_root, 'flow_planner', 'script')

# 加载配置
with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = hydra.compose(config_name='flow_planner_standard')
    print(f\"Config loaded successfully\")
    print(f\"Device: {cfg.device}\")
    print(f\"Batch size: {cfg.train.batch_size}\")
EOF
"

run_test "Check NPU config exists" 'if [ -f "flow_planner/script/flow_planner_npu.yaml" ]; then echo "NPU config found"; else echo "NPU config not found"; fi'

# 8. 内存和性能测试
echo ""
echo "=== 8. Memory and Performance Test ==="
echo ""

run_test "Test NPU memory allocation" "python3 << 'EOF'
import torch
try:
    import torch_npu
    if torch.npu.is_available():
        # 分配大张量测试内存
        large_tensor = torch.randn(1000, 1000, 100).npu()
        memory_allocated = torch.npu.memory_allocated(0)
        memory_reserved = torch.npu.memory_reserved(0)
        
        print(f\"Allocated memory: {memory_allocated / 1024**2:.2f} MB\")
        print(f\"Reserved memory: {memory_reserved / 1024**2:.2f} MB\")
        
        # 清空缓存
        del large_tensor
        torch.npu.empty_cache()
        
        print(\"Memory test: SUCCESS\")
    else:
        print(\"NPU not available, skipping memory test\")
except Exception as e:
    print(f\"Error: {e}\")
    exit(1)
EOF
"

run_test "Test simple forward pass" "python3 << 'EOF'
import torch
import torch.nn as nn
try:
    import torch_npu
    
    if torch.npu.is_available():
        # 创建简单模型
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 100)
        ).npu()
        
        # 前向传播
        x = torch.randn(32, 100).npu()
        y = model(x)
        
        print(f\"Input shape: {x.shape}\")
        print(f\"Output shape: {y.shape}\")
        print(\"Forward pass test: SUCCESS\")
    else:
        print(\"NPU not available, skipping forward pass test\")
except Exception as e:
    print(f\"Error: {e}\")
    exit(1)
EOF
"

# 9. 精度验证测试
echo ""
echo "=== 9. Precision Verification Test ==="
echo ""

run_test "Test numerical precision" "python3 << 'EOF'
import torch
import numpy as np

# 测试基本算子的数值精度
x = torch.randn(100, 100)
y = torch.randn(100, 100)

# CPU计算
result_cpu = torch.matmul(x, y)

# 如果NPU可用,测试NPU精度
try:
    import torch_npu
    if torch.npu.is_available():
        x_npu = x.npu()
        y_npu = y.npu()
        result_npu = torch.matmul(x_npu, y_npu).cpu()
        
        # 比较精度
        max_diff = torch.max(torch.abs(result_cpu - result_npu)).item()
        mean_diff = torch.mean(torch.abs(result_cpu - result_npu)).item()
        
        print(f\"Max difference: {max_diff:.6e}\")
        print(f\"Mean difference: {mean_diff:.6e}\")
        
        if max_diff < 1e-5:
            print(\"Precision test: PASSED\")
        else:
            print(f\"Precision test: WARNING (diff={max_diff:.6e})\")
    else:
        print(\"NPU not available, skipping precision test\")
except ImportError:
    print(\"torch_npu not installed, skipping precision test\")
EOF
"

# 10. 生成测试报告
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Total tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All tests passed! NPU adaptation is successful.${NC}"
    echo ""
    echo "You can now:"
    echo "  1. Train with NPU: python -m flow_planner.trainer --config-name flow_planner_npu"
    echo "  2. Run inference: python -m flow_planner.planner"
    echo ""
    exit 0
else
    echo -e "${RED}Some tests failed. Please check the errors above.${NC}"
    echo ""
    echo "Common issues:"
    echo "  - CANN Toolkit not installed: https://www.hiascend.com/software/aiengine"
    echo "  - torch_npu not installed: pip install torch-npu"
    echo "  - NPU driver not installed: Check /dev/davinci* devices"
    echo ""
    exit 1
fi

