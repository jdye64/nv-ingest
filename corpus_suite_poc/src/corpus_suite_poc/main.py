from __future__ import annotations

import typer
import uvicorn

app = typer.Typer(no_args_is_help=True)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8765, help="HTTP port"),
    reload: bool = typer.Option(False, help="Enable autoreload (dev)"),
) -> None:
    """Run the corpus HTTP service."""
    uvicorn.run(
        "corpus_suite_poc.agent_api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
