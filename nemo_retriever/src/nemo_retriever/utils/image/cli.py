# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from . import pdf_to_image, render

app = typer.Typer(help="Utilities for working with images (visualization, inspection, conversions)")
app.add_typer(render.app, name="render")
app.add_typer(pdf_to_image.app, name="pdf-to-image")
