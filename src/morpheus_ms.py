import logging

import click
from morpheus.config import Config

from pipeline import pipeline


@click.command()
@click.option("--batch-size", default=1, help="Pipeline batch size.")
@click.option("--num-threads", default=1, help="Number of threads to use.")
@click.option("--enable-monitor", is_flag=True, help="Enable monitoring.")
def run_pipeline(batch_size, num_threads, enable_monitor):
    """Runs the pipeline with the provided configuration."""
    logging.basicConfig(level=logging.INFO)

    config = Config()
    config.pipeline_batch_size = batch_size
    config.num_threads = num_threads
    config.enable_monitor = enable_monitor

    total_elapsed = pipeline(config)
    click.echo(f"Total time elapsed: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    run_pipeline()
