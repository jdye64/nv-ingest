# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from setuptools import find_packages
from setuptools import setup


def read_requirements(file_name):
    """Read a requirements file and return a list of its packages."""
    with open(file_name) as f:
        return f.read().splitlines()


# Specify your requirements files
requirements_files = [
    "requirements.txt",
    "util-requirements.txt",
    "test-requirements.txt",
]

# Read and combine requirements from all specified files
combined_requirements = []
for file in requirements_files:
    combined_requirements.extend(read_requirements(file))

combined_requirements = list(set(combined_requirements))

setup(
    name="nv_ingest",
    version="0.1.0",
    description="Python module supporting document ingestion",
    author="Devin Robison",
    author_email="drobison@nvidia.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=combined_requirements,
    classifiers=[],
    python_requires=">=3.10",
)
