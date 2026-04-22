"""TuiML Command-Line Interface (CLI).

This package provides a comprehensive set of command-line tools for 
building, evaluating, and managing machine learning workflows.
"""

import click
from tuiml import __version__

@click.group()
@click.version_option(version=__version__, prog_name="tuiml")
@click.pass_context
def cli(ctx):
    """
    TuiML - Modern Machine Learning CLI

    A Python-based ML framework with three levels of API.
    Use exact class names everywhere - no mappings, fully scalable!
    """
    pass

# Import commands
from tuiml.cli.commands import train, predict, evaluate, experiment, list_cmd, serve, setup, uninstall

# Register commands
cli.add_command(train.train)
cli.add_command(predict.predict)
cli.add_command(evaluate.evaluate)
cli.add_command(experiment.experiment)
cli.add_command(list_cmd.list_algorithms)
cli.add_command(serve.serve)
cli.add_command(setup.setup)
cli.add_command(uninstall.uninstall)

def main():
    """Main entry point for CLI."""
    cli(obj={})

if __name__ == "__main__":
    main()
