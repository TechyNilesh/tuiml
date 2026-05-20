"""`tuiml mcp` — run the TuiML MCP server in the foreground (for debugging).

Equivalent to invoking the `tuiml-mcp` binary directly, but exposed under
the unified `tuiml` CLI so users don't have to remember the second
executable name. Useful when an AI client launches `tuiml-mcp` via a
config entry and you want to inspect its stdout/stderr by hand.
"""
from __future__ import annotations

import click


@click.command()
def mcp() -> None:
    """Run the TuiML MCP server (stdio transport)."""
    # Import lazily so `tuiml --help` doesn't pay the import cost.
    from tuiml.agent.mcp.server import main as mcp_main
    mcp_main()
