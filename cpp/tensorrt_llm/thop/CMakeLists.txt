# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

if(NOT WIN32)
  # additional warnings
  #
  # Ignore overloaded-virtual warning. We intentionally change parameters of
  # some methods in derived class.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  if(WARNING_IS_ERROR)
    message(STATUS "Treating warnings as errors in GCC compilation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
else() # Windows
  # warning level 4
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

if(CONTEXT_ATTENTION_FMHA_CUBIN_INCLUDE_DIR)
  include_directories(BEFORE "${CONTEXT_ATTENTION_FMHA_CUBIN_INCLUDE_DIR}")
endif()

add_library(th_utils STATIC thUtils.cpp)
set_property(TARGET th_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET th_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(th_utils PUBLIC ${TORCH_LIBRARIES} ${CUBLAS_LIB}
                                      ${CURAND_LIB})

# TODO This does not compile with internal cutlass MOE gemm
add_library(
  th_common SHARED
  mlaPreprocessOp.cpp
  allgatherOp.cpp
  allreduceOp.cpp
  alltoallOp.cpp
  attentionOp.cpp
  causalConv1dOp.cpp
  convertSpecDecodingMaskToPackedMaskOp.cpp
  cuteDslMoeUtilsOp.cpp
  cutlassScaledMM.cpp
  cublasScaledMM.cpp
  cudaNvfp4MM.cpp
  cudaScaledMM.cpp
  dynamicDecodeOp.cpp
  fmhaPackMaskOp.cpp
  fp8Op.cpp
  fp8PerTensorScalingTrtllmGenGemm.cpp
  fp4Op.cpp
  fp4Gemm.cpp
  fp4GemmTrtllmGen.cpp
  fp8BatchedGemmTrtllmGen.cpp
  fp4Quantize.cpp
  fp4BatchedQuantize.cpp
  fp4xFp8GemmTrtllmGen.cpp
  fp8BlockScalingGemm.cpp
  fp8RowwiseGemm.cpp
  fp8Quantize.cpp
  dsv3FusedAGemmOp.cpp
  deepseekResidentDecodeOp.cpp
  fusedQKNormRopeOp.cpp
  fusedDiTQKNormRopeOp.cpp
  fusedDiTSplitQKNormRopeOp.cpp
  fusedDiTSplitNormOp.cpp
  fusedAddRMSNormQuant.cpp
  fusedActivationQuant.cpp
  fusedGatedRMSNormQuant.cpp
  fusedTopkSoftmax.cpp
  gatherTreeOp.cpp
  groupRmsNormOp.cpp
  helixPostProcessOp.cpp
  llama4MinLatency.cpp
  logitsBitmaskOp.cpp
  mambaConv1dOp.cpp
  moeOp.cpp
  moeUtilOp.cpp
  moeCommOp.cpp
  moeAlltoAllOp.cpp
  moeLoadBalanceOp.cpp
  moeAlignOp.cpp
  mxFp4BlockScaleMoe.cpp
  mxFp8Quantize.cpp
  fp8BlockScaleMoe.cpp
  fp8PerTensorScaleMoe.cpp
  fp4BlockScaleMoe.cpp
  noAuxTcOp.cpp
  fusedCatFp4Op.cpp
  fusedCatFp8Op.cpp
  fusedRopeCatFp4Op.cpp
  IndexerHisaNvfp4Op.cpp
  IndexerKCacheGatherOp.cpp
  IndexerKCacheScatterOp.cpp
  mlaBdrKvarnOp.cpp
  hisparseRequestTableOp.cpp
  hisparseSwapInPackedKvarnOp.cu
  hisparseKvarnHotReadOp.cpp
  hisparseTopkToBlocksOp.cpp
  kvarnGqaOp.cpp
  IndexerTopKOp.cpp
  SparseMlaDecodeKvarnHotOp.cpp
  SparseMlaDecodeNvfp4Op.cpp
  indexerXstepRecencyPatchOp.cpp
  ncclCommunicatorOp.cpp
  allocateOutput.cpp
  parallelDecodeKVCacheUpdateOp.cpp
  redrafterCurandOp.cpp
  reducescatterOp.cpp
  relativeAttentionBiasOp.cpp
  dsv3RouterGemmOp.cpp
  customMoeRoutingOp.cpp
  mamba2MTPSSMCacheOp.cpp
  selectiveScanOp.cpp
  userbuffersFinalizeOp.cpp
  userbuffersTensor.cpp
  virtualMemoryAllocator.cpp
  weightOnlyQuantGemm.cpp
  weightOnlyQuantOp.cpp
  specDecOp.cpp
  dynamicTreeOp.cpp
  loraOp.cpp
  finegrained_mixed_dtype_gemm_thop.cpp
  tinygemm2.cpp
  dsv3RopeOp.cpp
  convertReqIndexToGlobalOp.cpp
  indexerAffineReuseOp.cpp
  trtllmGenQKVProcessOp.cpp)

# Narrow proof target for the OP-TRT HiSparse native path. It compiles the real
# torch registrations and production CUDA kernels used by the HiSparse smoke
# tests without linking the full tensorrt_llm shared library and its unrelated
# generated CUTLASS/MoE targets. Production wheels still use th_common.
add_library(
  th_hisparse_smoke SHARED
  mlaBdrKvarnOp.cpp
  hisparseRequestTableOp.cpp
  hisparseSwapInPackedKvarnOp.cu
  hisparseKvarnHotReadOp.cpp
  hisparseTopkToBlocksOp.cpp
  SparseMlaDecodeKvarnHotOp.cpp
  ../common/assert.cpp
  ../common/logger.cpp
  ../common/stringUtils.cpp
  ../common/tllmException.cpp
  ../kernels/mlaBdrKvarnRecordKernels.cu
  ../kernels/hisparseTopkToBlocks.cu
  ../kernels/hisparseKvarnHotRead.cu
  ../kernels/flashMLA/sparse_mla_decode_kvarn_hot.cu)
set_property(TARGET th_hisparse_smoke PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET th_hisparse_smoke PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET th_hisparse_smoke PROPERTY CUDA_STANDARD 20)
set_property(TARGET th_hisparse_smoke PROPERTY CUDA_STANDARD_REQUIRED ON)
# The HiSparse hot-read decode kernel now uses the sm100 tcgen05/UMMA + TMEM machinery
# from nvfp4_sparse (kerutils device headers + cute/cutlass). Mirror the include paths +
# sm_100a architecture used by the flash_mla_nvfp4_sparse_src object library so the direct
# th_hisparse_smoke compile of sparse_mla_decode_kvarn_hot.cu resolves them and enables
# tcgen05 codegen.
target_include_directories(
  th_hisparse_smoke
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/flashMLA/nvfp4_sparse
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/flashMLA/nvfp4_sparse/kerutils/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/flashMLA/nvfp4_sparse/sm100/decode/head64_nvfp4
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/flashMLA
    $<TARGET_PROPERTY:cutlass_src,INTERFACE_INCLUDE_DIRECTORIES>)
# 100 (not "100a") so the helper emits arch=compute_100a,code=[sm_100a] with a REAL sm_100a
# cubin (matches flash_mla_nvfp4_sparse_src); "100a" alone would emit PTX-only and the
# driver could not load it ("no kernel image").
set_cuda_architectures(th_hisparse_smoke 100)
target_link_libraries(
  th_hisparse_smoke PRIVATE ${TORCH_LIBRARIES} th_utils ${Python3_LIBRARIES}
                            ${CUDA_DRV_LIB} ${CUBLAS_LIB} ${CURAND_LIB})
target_compile_options(
  th_hisparse_smoke
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
if(NOT WIN32)
  set_target_properties(
    th_hisparse_smoke
    PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/../../nvidia/nccl/lib"
               LINK_FLAGS "${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")
endif()

set_property(TARGET th_common PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(
  th_common PRIVATE ${TORCH_LIBRARIES} th_utils ${Python3_LIBRARIES}
                    ${SHARED_TARGET} pg_utils)

if(TARGET dsv3_min_latency_kernels)
  target_sources(th_common PRIVATE $<TARGET_OBJECTS:dsv3_min_latency_kernels>)
endif()

if(BUILD_DEEP_GEMM)
  set(TRTLLM_DEEP_GEMM_SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/deepgemm-src)
  target_compile_definitions(
    th_common PRIVATE TRTLLM_OPTRT_ENABLE_DEEP_GEMM_RESIDENT_INDEXER=1)
  target_include_directories(
    th_common
    PRIVATE ${TRTLLM_DEEP_GEMM_SOURCE_DIR}/csrc
            ${TRTLLM_DEEP_GEMM_SOURCE_DIR}/deep_gemm/include
            ${TRTLLM_DEEP_GEMM_SOURCE_DIR}/third-party/cutlass/include
            ${TRTLLM_DEEP_GEMM_SOURCE_DIR}/third-party/fmt/include)
  target_link_libraries(th_common PRIVATE CUDA::cuda_driver CUDA::cudart)
endif()

if(USING_OSS_CUTLASS_LOW_LATENCY_GEMM)
  target_compile_definitions(th_common
                             PUBLIC "USING_OSS_CUTLASS_LOW_LATENCY_GEMM")
endif()

if(USING_OSS_CUTLASS_FP4_GEMM)
  target_compile_definitions(th_common PUBLIC USING_OSS_CUTLASS_FP4_GEMM)
endif()

if(USING_OSS_CUTLASS_MOE_GEMM)
  target_compile_definitions(th_common PUBLIC USING_OSS_CUTLASS_MOE_GEMM)
endif()

if(ENABLE_CUBLASLT_FP4_GEMM)
  target_sources(th_common PRIVATE cublasFp4ScaledMM.cpp)
  target_compile_definitions(th_common PUBLIC ENABLE_CUBLASLT_FP4_GEMM)
  target_link_libraries(th_common PRIVATE ${CUBLASLT_LIB})
endif()

if(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
  target_sources(th_common PRIVATE fusedGemmAllreduceOp.cpp)
  target_compile_definitions(th_common PUBLIC USING_OSS_CUTLASS_ALLREDUCE_GEMM)
endif()

if(ENABLE_MULTI_DEVICE)
  target_include_directories(th_common PUBLIC ${MPI_C_INCLUDE_DIRS})
  target_link_libraries(th_common PRIVATE ${MPI_C_LIBRARIES} ${NCCL_LIB})
endif()

if(NOT WIN32)
  set_target_properties(
    th_common PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/..;$ORIGIN/../../nvidia/nccl/lib")
  set_target_properties(
    th_common PROPERTIES LINK_FLAGS "${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")
else()
  target_link_libraries(th_common PRIVATE context_attention_src)
endif()
