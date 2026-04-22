"""Uninstall Command — remove TuiML's MCP wiring from every detected AI client.

This command is the inverse of ``tuiml setup``. It does NOT remove the
``tuiml`` package itself (use ``uv tool uninstall tuiml`` or ``pip uninstall
tuiml`` for that). It only unwires the MCP server entries and the Claude Code
skill file that ``tuiml setup`` wrote.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import click

from tuiml.cli.commands.setup import (
    ALL_CLIENT_IDS,
    C,
    backup_file,
    client_specs,
    confirm,
    describe_target,
    error,
    info,
    section,
    success,
    warn,
    _get_nested,
    _set_nested,
)


SERVER_NAME = "tuiml"


def _banner() -> None:
    click.echo()
    click.echo(f"  {C.BOLD}{C.BLUE}TuiML Uninstall Wizard{C.RESET}")
    click.echo(f"  {C.DIM}Remove TuiML from your AI agents{C.RESET}")
    click.echo()


# ---------------------------------------------------------------------------
# Per-kind removers
# ---------------------------------------------------------------------------

def remove_json_entry(config_path: Path, key: str) -> tuple[bool, str]:
    """Remove the ``tuiml`` entry from a JSON config under ``key`` (supports dotted keys)."""
    if not config_path.exists():
        return False, "config file not present"
    try:
        data = json.loads(config_path.read_text() or "{}")
    except json.JSONDecodeError as exc:
        return False, f"existing config is not valid JSON: {exc}"

    block = _get_nested(data, key)
    if not isinstance(block, dict) or SERVER_NAME not in block:
        return False, "no tuiml entry found"

    backup_file(config_path)
    del block[SERVER_NAME]
    _set_nested(data, key, block)
    config_path.write_text(json.dumps(data, indent=2))
    return True, "removed tuiml entry"


def remove_toml_entry(config_path: Path) -> tuple[bool, str]:
    """Strip the ``[mcp_servers.tuiml]`` block from a TOML config (Codex CLI style)."""
    if not config_path.exists():
        return False, "config file not present"

    text = config_path.read_text()
    header = f"[mcp_servers.{SERVER_NAME}]"
    if header not in text:
        return False, "no tuiml entry found"

    backup_file(config_path)
    pattern = re.compile(
        rf"\[mcp_servers\.{re.escape(SERVER_NAME)}\][^\[]*",
        re.MULTILINE,
    )
    cleaned = pattern.sub("", text).rstrip() + "\n"
    config_path.write_text(cleaned)
    return True, "removed [mcp_servers.tuiml] block"


def remove_skill_dir(skills_dir: Path) -> tuple[bool, str]:
    """Delete the bundled ``tuiml`` skill directory (Claude Code)."""
    target_dir = skills_dir / "tuiml"
    if not target_dir.exists():
        return False, "skill directory not present"
    shutil.rmtree(target_dir)
    return True, f"deleted {target_dir}"


def print_yaml_instructions(spec: dict) -> tuple[bool, str]:
    """Print manual uninstall instructions for YAML-config clients (Goose)."""
    info(f"  Edit {spec['config']} and remove the `extensions.tuiml` block.")
    return False, "manual step (YAML config not auto-edited)"


def unconfigure(spec: dict) -> tuple[bool, str]:
    kind = spec["kind"]
    if kind == "json-mcp":
        return remove_json_entry(spec["config"], "mcpServers")
    if kind == "json-key":
        return remove_json_entry(spec["config"], spec["key"])
    if kind == "toml-mcp":
        return remove_toml_entry(spec["config"])
    if kind == "skill":
        return remove_skill_dir(spec["skills_dir"])
    if kind == "yaml-instructions":
        return print_yaml_instructions(spec)
    return False, f"unknown client kind: {kind}"


def prompt_mode(default: str = "auto") -> str:
    """Ask whether to unwire every detected client at once or one by one."""
    click.echo()
    click.echo(f"  {C.BOLD}Mode:{C.RESET}")
    click.echo(f"    [{C.GREEN}a{C.RESET}] Auto   — remove TuiML from every detected client")
    click.echo(f"    [{C.YELLOW}m{C.RESET}] Manual — ask for each client individually")
    click.echo(f"    [{C.DIM}q{C.RESET}] Quit")
    choice = click.prompt("  Choose", default=default, show_default=True).strip().lower()
    if choice.startswith("q"):
        return "quit"
    if choice.startswith("m"):
        return "manual"
    return "auto"


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------

@click.command("uninstall")
@click.option("--yes", "-y", "assume_yes", is_flag=True,
              help="Auto mode: unwire every detected client without prompting.")
@click.option("--manual", "force_manual", is_flag=True,
              help="Manual mode: ask per-client (skip the Auto/Manual menu).")
@click.option("--client", "clients", multiple=True,
              help="Unwire only the specified client(s). Repeatable. "
                   "Run 'tuiml setup --list' to see valid IDs.")
def uninstall(assume_yes: bool, force_manual: bool, clients: tuple[str, ...]) -> None:
    """Remove TuiML from your AI agents (inverse of ``tuiml setup``).

    Scans every supported client, and for each one that currently has a
    ``tuiml`` MCP entry (or skill), removes it. Each config file is backed up
    before being modified.

    This command does NOT uninstall the ``tuiml`` Python package itself. After
    running it, finish the job with:

        uv tool uninstall tuiml    # if installed via `uv tool install`
        pip uninstall tuiml        # if installed via pip
    """
    _banner()

    info("Scanning AI clients for TuiML wiring ...")

    # Consider every known client (not just detected ones) — a config file may
    # still have a stale tuiml entry even if the detect path is gone.
    all_specs = client_specs()

    if clients:
        unknown = [c for c in clients if c not in ALL_CLIENT_IDS]
        if unknown:
            error(f"Unknown client(s): {', '.join(unknown)}")
            info(f"Valid IDs: {', '.join(ALL_CLIENT_IDS)}")
            sys.exit(1)
        all_specs = [s for s in all_specs if s["id"] in clients]

    if assume_yes:
        mode = "auto"
    elif force_manual:
        mode = "manual"
    else:
        mode = prompt_mode(default="auto")

    if mode == "quit":
        info("Cancelled — no changes made.")
        click.echo()
        return

    auto = (mode == "auto")

    section("Removing wiring:")
    changes_made = 0
    for spec in all_specs:
        if not auto and not confirm(f"Unwire {spec['name']}?", default=True):
            info(f"  Skipped {spec['name']}")
            continue

        try:
            changed, reason = unconfigure(spec)
        except Exception as exc:
            error(f"  {spec['name']}: {exc}")
            continue

        if changed:
            success(f"  {spec['name']}: {reason}")
            changes_made += 1
        else:
            info(f"  {spec['name']:22} {C.DIM}({reason}){C.RESET}")

    section("Done.")
    if changes_made:
        info(f"Removed TuiML from {changes_made} client(s). Restart any running clients.")
    else:
        info("No client wiring was found to remove.")
    click.echo()

    click.echo(f"  {C.DIM}To remove the Python package as well, run one of:{C.RESET}")
    click.echo(f"    {C.BOLD}uv tool uninstall tuiml{C.RESET}     {C.DIM}# if installed via `uv tool install`{C.RESET}")
    click.echo(f"    {C.BOLD}pip uninstall tuiml{C.RESET}         {C.DIM}# if installed via pip{C.RESET}")
    click.echo()
