"""`tuiml info` — show installation details and check for updates.

Thin CLI wrapper around the existing `execute_system_info` MCP tool so
both surfaces stay in sync.
"""
from __future__ import annotations

import json as _json

import click


@click.command()
@click.option(
    "--no-check",
    is_flag=True,
    help="Skip the PyPI latest-version check (faster, no network).",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit raw JSON instead of the human-readable table.",
)
def info(no_check: bool, as_json: bool) -> None:
    """Show TuiML version, install method, and update status."""
    from tuiml.agent.tools import execute_system_info

    result = execute_system_info(check_latest=not no_check)

    if as_json:
        click.echo(_json.dumps(result, indent=2, default=str))
        return

    if result.get("status") != "success":
        click.echo(f"✗ {result.get('error', 'unknown error')}", err=True)
        raise click.exceptions.Exit(1)

    rows = [
        ("Version",         result.get("version")),
        ("Install method",  result.get("install_method")),
        ("Package path",    result.get("package_path")),
        ("Python",          f"{result.get('python_version')}  ({result.get('python_executable')})"),
        ("Platform",        result.get("platform")),
    ]
    if "latest_version" in result:
        latest = result["latest_version"]
        flag = "" if not result.get("update_available") else "  ← update available"
        rows.append(("Latest on PyPI", f"{latest}{flag}"))
        if result.get("update_available"):
            rows.append(("Upgrade hint",   result.get("upgrade_hint")))
    elif "latest_version_error" in result:
        rows.append(("Latest on PyPI", f"(check failed: {result['latest_version_error']})"))

    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        click.echo(f"  {k.ljust(width)}  {v}")
