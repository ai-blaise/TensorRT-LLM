# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..logger import logger

IS_CUBLASLT_AVAILABLE = False

# Check cuBLASLt availability
try:
    import torch

    # Check if CublasLtFP4GemmRunner (used by nvfp4_gemm_cublaslt) is available
    try:
        trtllm_classes = getattr(torch.classes, 'trtllm', None)
        has_runner = (
            trtllm_classes is not None
            and getattr(trtllm_classes, 'CublasLtFP4GemmRunner', None)
            is not None)
    except (AttributeError, RuntimeError):
        has_runner = False
    if has_runner:
        logger.info(f"cuBLASLt FP4 GEMM is available")
        IS_CUBLASLT_AVAILABLE = True
except ImportError:
    pass
