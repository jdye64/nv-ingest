# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``retriever config`` CLI: inspect, serialize, and document configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    help=(
        "Inspect and document NeMo Retriever configuration. The configuration "
        "is the single source of truth used by the service, harnesses, tools, "
        "and web pages."
    ),
    no_args_is_help=True,
)


@app.command()
def show(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to a unified config file."),
    remote: Optional[str] = typer.Option(None, "--remote", help="Base URL of a cluster config server."),
    fmt: str = typer.Option("yaml", "--format", "-f", help="Output format: yaml or json."),
    no_env: bool = typer.Option(False, "--no-env", help="Ignore NEMO_RETRIEVER_* environment variables."),
    show_secrets: bool = typer.Option(False, "--show-secrets", help="Do not redact secret values."),
) -> None:
    """Load configuration from all layers and print it."""
    from nemo_retriever.config import ConfigService

    service = ConfigService.load(config, remote_url=remote, use_env=not no_env)
    typer.echo(f"# source: {service.source}", err=True)
    typer.echo(service.serialize(fmt, redact_secrets=not show_secrets))


@app.command()
def save(
    output: str = typer.Argument(..., help="Destination file (.yaml or .json)."),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to a unified config file."),
    no_env: bool = typer.Option(False, "--no-env", help="Ignore NEMO_RETRIEVER_* environment variables."),
    show_secrets: bool = typer.Option(False, "--show-secrets", help="Do not redact secret values."),
) -> None:
    """Load configuration and write it to a file."""
    from nemo_retriever.config import ConfigService

    service = ConfigService.load(config, use_env=not no_env)
    path = service.save(output, redact_secrets=not show_secrets)
    typer.echo(f"Wrote configuration to {path}")


@app.command()
def schema(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write JSON Schema to this file."),
) -> None:
    """Print the JSON Schema for the unified configuration object."""
    from nemo_retriever.config import NeMoRetrieverConfig

    data = json.dumps(NeMoRetrieverConfig.model_json_schema(), indent=2)
    if output:
        Path(output).write_text(data)
        typer.echo(f"Wrote JSON Schema to {output}")
    else:
        typer.echo(data)


@app.command()
def docs(
    output: str = typer.Option(
        "fern/docs/pages/configuration.mdx",
        "--output",
        "-o",
        help="Destination page.",
    ),
    fmt: str = typer.Option("fern", "--format", "-f", help="Doc format: fern or markdown."),
    title: str = typer.Option("Configuration Reference", "--title", help="Page title."),
) -> None:
    """Generate the extensive, self-documenting configuration reference page."""
    from nemo_retriever.config import generate_config_docs

    path = generate_config_docs(output, fmt=fmt, title=title)
    typer.echo(f"Generated configuration docs at {path}")


@app.command()
def categories() -> None:
    """Summarize configuration categories and how many settings each covers."""
    from nemo_retriever.config import (
        ConfigCategory,
        NeMoRetrieverConfig,
        build_catalog,
        fields_by_category,
    )

    grouped = fields_by_category(build_catalog(NeMoRetrieverConfig))
    for category in ConfigCategory:
        count = len(grouped.get(category, []))
        typer.echo(f"{category.value:16s} {count:4d}  {category.summary}")


if __name__ == "__main__":  # pragma: no cover
    app()
