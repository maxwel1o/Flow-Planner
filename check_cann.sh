#!/bin/bash
# CANN路径检测和修复脚本

echo "=========================================="
echo "CANN Path Detection Script"
echo "=========================================="

# 检查可能的CANN安装路径
POSSIBLE_PATHS=(
    "/usr/local/Ascend/cann-8.5.1"
    "/usr/local/Ascend/ascend-toolkit/latest"
    "/usr/local/Ascend/ascend-toolkit/8.5.1"
    "/home/Ascend/cann-8.5.1"
    "/opt/Ascend/cann-8.5.1"
    "$HOME/Ascend/cann-8.5.1"
)

echo ""
echo "Checking possible CANN installation paths..."
echo ""

FOUND_PATH=""
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "✓ Found: $path"
        FOUND_PATH="$path"
        break
    else
        echo "✗ Not found: $path"
    fi
done

if [ -z "$FOUND_PATH" ]; then
    echo ""
    echo "ERROR: CANN installation not found!"
    echo ""
    echo "Please check:"
    echo "1. Is CANN installed?"
    echo "2. What is the installation path?"
    echo ""
    echo "Try these commands to find CANN:"
    echo "  find /usr -name 'ascend-toolkit' -type d 2>/dev/null"
    echo "  find /opt -name 'cann-*' -type d 2>/dev/null"
    echo "  ls -la /usr/local/Ascend/"
    exit 1
fi

echo ""
echo "Using CANN path: $FOUND_PATH"
echo ""

# 检查目录结构
echo "Checking directory structure..."
echo ""

# 可能的TBE路径
TBE_PATHS=(
    "$FOUND_PATH/toolkit/python/site-packages"
    "$FOUND_PATH/python/site-packages"
    "$FOUND_PATH/opp/built-in/op_impl/ai_core/tbe"
    "$FOUND_PATH/ascend-toolkit/latest/toolkit/python/site-packages"
)

TBE_PATH=""
for path in "${TBE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "✓ TBE path found: $path"
        TBE_PATH="$path"
        break
    else
        echo "✗ Not found: $path"
    fi
done

if [ -z "$TBE_PATH" ]; then
    echo ""
    echo "ERROR: TBE path not found in CANN installation"
    echo ""
    echo "Listing CANN directory structure:"
    find "$FOUND_PATH" -maxdepth 3 -type d | head -20
    exit 1
fi

echo ""
echo "=========================================="
echo "Generating environment configuration..."
echo "=========================================="

# 生成正确的环境配置
cat > npu_env_fixed.sh << EOF
#!/bin/bash
# NPU环境配置 - 自动生成

# CANN路径
export ASCEND_TOOLKIT_HOME=$FOUND_PATH

# TBE路径
export TBE_IMPL_PATH=$FOUND_PATH/toolkit/ops/built-in/op_impl/ai_core/tbe
export ASCEND_OPP_PATH=$FOUND_PATH/opp
export ASCEND_OPP_BUILT_IN=\$ASCEND_OPP_PATH/built-in

# Python路径
export PYTHONPATH=$TBE_PATH:\$PYTHONPATH
export PYTHONPATH=$TBE_PATH/tbe:\$PYTHONPATH
export PYTHONPATH=$TBE_PATH/tbe/ops:\$PYTHONPATH

# 其他环境变量
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TBE_PARALLEL_COMPILER=1

echo "NPU environment configured:"
echo "  ASCEND_TOOLKIT_HOME=\$ASCEND_TOOLKIT_HOME"
echo "  TBE_PATH=$TBE_PATH"
EOF

chmod +x npu_env_fixed.sh

echo ""
echo "✓ Configuration generated: npu_env_fixed.sh"
echo ""
echo "Next steps:"
echo "  1. source npu_env_fixed.sh"
echo "  2. python3 -c 'import tbe; print(\"TBE OK\")'"
echo "  3. bash verify_npu.sh"

