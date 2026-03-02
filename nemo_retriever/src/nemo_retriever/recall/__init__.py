# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Recall evaluation utilities and CLI.
"""

from .__main__ import app
from .core import RecallConfig, evaluate_recall, gold_to_doc_page, hit_key_and_distance, is_hit_at_k

__all__ = [
    "app",
    "RecallConfig",
    "evaluate_recall",
    "gold_to_doc_page",
    "hit_key_and_distance",
    "is_hit_at_k",
]
