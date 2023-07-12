#!/bin/bash

# clear
# set -x

# log dir
ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

# rm -rf log

# export AMD_LOG_LEVEL=4
# export HIP_VISIBLE_DEVICES=0
# export HIP_LAUNCH_BLOCKING=1


# export MLIR_ENABLE_DUMP=1
# export LLVM_IR_ENABLE_DUMP=1
# export AMDGCN_ENABLE_DUMP=1

# UNIT_TEST="test_amd.py"
# UNIT_TEST="test_jax.py"
# UNIT_TEST="tests"
# UNIT_TEST="tests/triton_test.py"
# UNIT_TEST="tests/triton_call_test.py"
# UNIT_TEST="tests/pallas_test.py"
# UNIT_TEST="tests/pallas_test.py::PallasCallTest::test_softmax_1_1_128_float16" # fails
# UNIT_TEST="tests/pallas_test.py::PallasCallInterpreterTest::test_softmax_1_1_128_float16" # passes
# UNIT_TEST="tests/pallas_test.py::PallasCallTest::test_softmax_1_1_128_float32" # fails
# UNIT_TEST="tests/pallas_test.py::PallasCallInterpreterTest::test_softmax_1_1_128_float32" # passes
# UNIT_TEST="tests/pallas_test.py::PallasCallTest::test_softmax_1_1_2_float16"
# UNIT_TEST="tests/pallas_test.py::PallasCallTest::test_softmax_1_1_2_float32"
# UNIT_TEST="tests/pallas_test.py::PallasCallTest::test_softmax_1_129_256_float32"
# UNIT_TEST="tests/pallas_test.py::PallasCallInterpreterTest::test_softmax"
# UNIT_TEST="tests/pallas_test.py::PallasCallInterpreterTest::test_matmul"
# UNIT_TEST="tests/pallas_test.py::PallasCallInterpreterTest::test_matmul_m_1024_n_1024_k_512_dtype_float16_bm_128_bn_128_bk_32_gm_8"
# UNIT_TEST="tests/pallas_test.py::PallasCallTest::test_matmul_m_1024_n_1024_k_512_dtype_float32_bm_128_bn_128_bk_32_gm_8" #fails
# UNIT_TEST="tests/pallas_test.py::PallasCallInterpreterTest::test_matmul_m_1024_n_1024_k_512_dtype_float32_bm_128_bn_128_bk_32_gm_8" #passes

# UNIT_TEST="tests/pallas_test.py::PallasCallTest::test_matmul_m_512_n_512_k_512_dtype_float32_bm_64_bn_64_bk_32_gm_8" # works
# UNIT_TEST="tests/pallas_test.py::PallasCallTest::test_matmul_m_512_n_512_k_512_dtype_float32_bm_64_bn_128_bk_32_gm_8" # fails
# UNIT_TEST="tests/pallas_test.py::PallasCallTest::test_matmul"
# UNIT_TEST="tests/pallas_test.py::PallasCallTest::test_add_matrix_block_spec"
# UNIT_TEST="tests/triton_call_test.py::TritonKernelCallTest::test_add0"
# UNIT_TEST="tests/triton_call_test.py::TritonKernelCallTest::test_matmul0"
# UNIT_TEST="tests/triton_test.py::TritonTest::test_add_kernel"
# UNIT_TEST="tests/triton_test.py::TritonTest::test_tanh_kernel"

# FILTER="-k test_softmax"
# FILTER="-k test_matmul"
# FILTER="-k test_matmul_block"
# FILTER="-k test_matmul_m"
# FILTER="-k test_matmul_m_512_n_512_k_512_dtype_float32"
# FILTER="-k test_fused"
# FILTER="-k test_rms"
# FILTER="-k test_jvp"
# FILTER="-k test_vmap"
# FILTER="-k test_fused"

LOG_FILE_NAME=$(basename $UNIT_TEST)

# OUTPUT_MODE="--capture=tee-sys -v"
OUTPUT_MODE="--capture=tee-sys -vv"

# check for backtrace
if [ "$1" == "backtrace" ]; then
    sudo apt install gdb -y

    COMMAND="-m pytest $OUTPUT_MODE $UNIT_TEST"
    gdb python \
        -ex "set pagination off" \
        -ex "run $COMMAND" \
        -ex "backtrace" \
        -ex "set confirm off" \
        -ex "q" \
        2>&1 | tee $LOG_DIR/${LOG_FILE_NAME}_backtrace.log

else
    pytest $OUTPUT_MODE -rfs --verbose "$UNIT_TEST" $FILTER 2>&1 | tee $LOG_DIR/${LOG_FILE_NAME}.log
fi
