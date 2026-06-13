#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
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
#

file(GLOB_RECURSE SRC_CPP *.cpp)
file(GLOB_RECURSE SRC_CU *.cu)
# SM100 sparse sources are built by the dedicated
# flash_mla_nvfp4_sparse_src target; exclude them from this SM90 glob.
list(FILTER SRC_CU EXCLUDE REGEX "nvfp4|sparse_mla_decode_kvarn_hot\\.cu")

add_library(flash_mla_src OBJECT ${SRC_CPP} ${SRC_CU})
set_property(TARGET flash_mla_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET flash_mla_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_cuda_architectures(flash_mla_src 90)


set(NVFP4_SPARSE_SRC_CU
    sparse_mla_decode_kvarn_hot.cu
    sparse_mla_decode_nvfp4.cu
    nvfp4_sparse/sm100/decode/head64_nvfp4/instantiations/v32.cu
    nvfp4_sparse/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
    nvfp4_sparse/smxx/decode/combine/combine.cu)

add_library(flash_mla_nvfp4_sparse_src OBJECT ${NVFP4_SPARSE_SRC_CU})
set_property(TARGET flash_mla_nvfp4_sparse_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET flash_mla_nvfp4_sparse_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET flash_mla_nvfp4_sparse_src PROPERTY CUDA_STANDARD 20)
set_property(TARGET flash_mla_nvfp4_sparse_src PROPERTY CUDA_STANDARD_REQUIRED ON)
target_include_directories(
  flash_mla_nvfp4_sparse_src
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/nvfp4_sparse
    ${CMAKE_CURRENT_SOURCE_DIR}/nvfp4_sparse/kerutils/include
    ${CMAKE_CURRENT_SOURCE_DIR}/nvfp4_sparse/sm100/decode/head64_nvfp4
    ${CMAKE_CURRENT_SOURCE_DIR}/nvfp4_sparse/smxx/decode/get_decoding_sched_meta
    ${CMAKE_CURRENT_SOURCE_DIR}/nvfp4_sparse/smxx/decode/combine
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:${INTERNAL_CUTLASS_KERNELS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:cutlass_src,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(
  flash_mla_nvfp4_sparse_src
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-D_USE_MATH_DEFINES -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math --ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage>
)
set_cuda_architectures(flash_mla_nvfp4_sparse_src 100)
