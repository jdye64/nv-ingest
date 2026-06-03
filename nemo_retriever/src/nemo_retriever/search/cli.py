# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typer sub-application for ``retriever search``."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer

app = typer.Typer(help="NeMo Retriever Search — minimal search UI over the retriever service.")


@app.command("start")
def start_command(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address."),
    port: int = typer.Option(8200, "--port", help="Bind port."),
    service_url: str = typer.Option(
        "http://localhost:7670",
        "--service-url",
        help="Retriever service base URL.",
        envvar="NEMO_RETRIEVER_SERVICE_URL",
    ),
    vectordb_url: str = typer.Option(
        "http://localhost:7671",
        "--vectordb-url",
        help="VectorDB service URL for corpus status.",
        envvar="NEMO_RETRIEVER_VECTORDB_URL",
    ),
    api_token: Optional[str] = typer.Option(
        None,
        "--api-token",
        help="Bearer token for retriever service requests.",
        envvar="NEMO_RETRIEVER_API_TOKEN",
    ),
) -> None:
    """Launch the NeMo Retriever Search web app."""
    import uvicorn

    from nemo_retriever.search.app import create_app
    from nemo_retriever.search.config import load_config
    from nemo_retriever.search.services.service_client import SERVICE_START_HINT, ServiceClient, ServiceUnavailableError

    config = load_config(
        service_url=service_url,
        vectordb_url=vectordb_url,
        api_token=api_token,
        host=host,
        port=port,
    )

    async def _preflight() -> None:
        client = ServiceClient(config)
        await client.get_service_health()
        vdb = await client.get_vectordb_health()
        if vdb is None:
            typer.echo(
                "Warning: VectorDB health check failed — corpus status may be unavailable. "
                "Ensure vectordb is running and --vectordb-url is correct.",
                err=True,
            )
        elif not vdb.get("table_exists") or int(vdb.get("total_rows") or 0) == 0:
            typer.echo(
                "Warning: No ingested documents found yet. Upload via the + button or ingest through the service.",
                err=True,
            )

    try:
        asyncio.run(_preflight())
    except ServiceUnavailableError as exc:
        typer.echo(f"Error: {exc}", err=True)
        typer.echo(SERVICE_START_HINT, err=True)
        raise typer.Exit(1) from exc

    application = create_app(config)
    typer.echo(f"Starting NeMo Retriever Search at http://{host}:{port}")
    typer.echo(f"Retriever service: {config.service_url}")
    uvicorn.run(application, host=host, port=port, log_level="info")


def main() -> None:
    app()
