# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from nv_ingest.schemas.otel_tracer_schema import OpenTelemetryTracerSchema


def test_otel_tracer_schema_defaults():
    schema = OpenTelemetryTracerSchema()
    assert schema.raise_on_failure is False, "Default value for raise_on_failure should be False."


def test_otel_tracer_schema_custom_values():
    schema = OpenTelemetryTracerSchema(raise_on_failure=True)
    assert schema.raise_on_failure is True, "Custom value for raise_on_failure should be respected."
