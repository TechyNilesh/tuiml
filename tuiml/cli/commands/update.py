"""`tuiml update` — upgrade tuiml to the latest PyPI version.

Thin CLI wrapper around the existing `execute_self_update` MCP tool.
Refuses to upgrade editable / dev checkouts (use `git pull` instead).
"""
from __future__ import annotations

import json as _json

import click


@click.command()
@click.option(
    "--target", "target_version",
    metavar="VERSION",
    help="Install a specific version (e.g. 0.1.4). Defaults to latest.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the command that would run; don't actually upgrade.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit raw JSON instead of the human-readable summary.",
)
def update(target_version: str, dry_run: bool, as_json: bool) -> None:
    """Upgrade TuiML using the install method that brought it in."""
    from tuiml.agent.tools import execute_self_update

    result = execute_self_update(target_version=target_version, dry_run=dry_run)

    if as_json:
        click.echo(_json.dumps(result, indent=2, default=str))
        raise click.exceptions.Exit(0 if result.get("status") == "success" else 1)

    if result.get("status") != "success":
        click.echo(f"✗ {result.get('error', 'upgrade failed')}", err=True)
        if "command" in result:
            click.echo(f"  command: {' '.join(result['command'])}", err=True)
        raise click.exceptions.Exit(1)

    if result.get("dry_run"):
        click.echo(f"would run: {' '.join(result['command'])}")
        click.echo(f"install method: {result['install_method']}")
        return

    before = result.get("previous_version", "?")
    after  = result.get("version", "?")
    click.echo(f"✓ TuiML upgraded: {before} → {after}")
    if result.get("install_method"):
        click.echo(f"  via: {result['install_method']}")
