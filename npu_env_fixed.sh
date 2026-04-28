#!/bin/bash
# NPU环境配置 - 自动生成

# CANN路径
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.1

# TBE路径
export TBE_IMPL_PATH=/usr/local/Ascend/cann-8.5.1/toolkit/ops/built-in/op_impl/ai_core/tbe
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-8.5.1/opp
export ASCEND_OPP_BUILT_IN=$ASCEND_OPP_PATH/built-in

# Python路径
export PYTHONPATH=/usr/local/Ascend/cann-8.5.1/python/site-packages:$PYTHONPATH
export PYTHONPATH=/usr/local/Ascend/cann-8.5.1/python/site-packages/tbe:$PYTHONPATH
export PYTHONPATH=/usr/local/Ascend/cann-8.5.1/python/site-packages/tbe/ops:$PYTHONPATH

# 其他环境变量
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TBE_PARALLEL_COMPILER=1

echo "NPU environment configured:"
echo "  ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME"
echo "  TBE_PATH=/usr/local/Ascend/cann-8.5.1/python/site-packages"
