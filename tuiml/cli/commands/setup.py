"""Setup Command - Interactive wizard to connect TuiML to AI agents."""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

class C:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


def banner() -> None:
    click.echo()
    click.echo(f"{C.BOLD}{C.BLUE}  TuiML Setup Wizard{C.RESET}")
    click.echo(f"{C.DIM}  Connect TuiML to your AI agents{C.RESET}")
    click.echo()


def info(msg: str) -> None:
    click.echo(f"{C.DIM}·{C.RESET} {msg}")


def success(msg: str) -> None:
    click.echo(f"{C.GREEN}✓{C.RESET} {msg}")


def warn(msg: str) -> None:
    click.echo(f"{C.YELLOW}!{C.RESET} {msg}")


def error(msg: str) -> None:
    click.echo(f"{C.RED}✗{C.RESET} {msg}")


def section(title: str) -> None:
    click.echo()
    click.echo(f"{C.BOLD}{title}{C.RESET}")


def confirm(prompt: str, default: bool = True, assume_yes: bool = False) -> bool:
    if assume_yes:
        return True
    return click.confirm(f"  {prompt}", default=default)


# ---------------------------------------------------------------------------
# Client registry
# ---------------------------------------------------------------------------
#
# Each client spec has:
#   id        : short slug used by --client flag
#   name      : human-readable label
#   detect    : Path that, if it exists, indicates the client is installed
#   kind      : "json-mcp"  (JSON config with mcpServers key)
#                "json-key" (JSON config with custom key, eg Zed)
#                "toml-mcp" (TOML config, eg OpenAI Codex CLI)
#                "skill"    (drop SKILL.md in a skills directory)
#   config    : Path to the config file (for json/toml)
#   key       : Top-level config key (default "mcpServers", overridable)
#   skills_dir: Path to skills directory (for skill kind)

HOME = Path.home()


def _xdg_config(name: str) -> Path:
    """Return $XDG_CONFIG_HOME/name (or ~/.config/name)."""
    import os
    base = Path(os.environ.get("XDG_CONFIG_HOME", HOME / ".config"))
    return base / name


def _platform_app_dir(macos: str, linux: str, windows: str) -> Path:
    """Return the per-platform application support directory."""
    if sys.platform == "darwin":
        return HOME / "Library" / "Application Support" / macos
    if sys.platform == "win32":
        return HOME / "AppData" / "Roaming" / windows
    # linux / wsl
    return _xdg_config(linux)


def client_specs() -> list[dict]:
    """Return all known MCP client specs."""
    return [
        # ----- Claude Desktop ----------------------------------------------
        {
            "id": "claude-desktop",
            "name": "Claude Desktop",
            "kind": "json-mcp",
            "detect": _platform_app_dir("Claude", "Claude", "Claude"),
            "config": _platform_app_dir("Claude", "Claude", "Claude") / "claude_desktop_config.json",
        },
        # ----- Claude Code (skill file) ------------------------------------
        {
            "id": "claude-code",
            "name": "Claude Code",
            "kind": "skill",
            "detect": HOME / ".claude",
            "skills_dir": HOME / ".claude" / "skills",
        },
        # ----- ChatGPT Desktop ---------------------------------------------
        # OpenAI added MCP support in 2026; config layout mirrors Claude Desktop.
        {
            "id": "chatgpt-desktop",
            "name": "ChatGPT Desktop",
            "kind": "json-mcp",
            "detect": _platform_app_dir("ChatGPT", "ChatGPT", "ChatGPT"),
            "config": _platform_app_dir("ChatGPT", "ChatGPT", "ChatGPT") / "mcp_config.json",
        },
        # ----- Perplexity Desktop (Comet) ----------------------------------
        {
            "id": "perplexity",
            "name": "Perplexity Desktop",
            "kind": "json-mcp",
            "detect": _platform_app_dir("Perplexity", "Perplexity", "Perplexity"),
            "config": _platform_app_dir("Perplexity", "Perplexity", "Perplexity") / "mcp_config.json",
        },
        # ----- OpenAI Codex CLI --------------------------------------------
        {
            "id": "codex",
            "name": "OpenAI Codex CLI",
            "kind": "toml-mcp",
            "detect": HOME / ".codex",
            "config": HOME / ".codex" / "config.toml",
        },
        # ----- Cursor ------------------------------------------------------
        {
            "id": "cursor",
            "name": "Cursor",
            "kind": "json-mcp",
            "detect": HOME / ".cursor",
            "config": HOME / ".cursor" / "mcp.json",
        },
        # ----- Windsurf (Codeium) ------------------------------------------
        {
            "id": "windsurf",
            "name": "Windsurf",
            "kind": "json-mcp",
            "detect": HOME / ".codeium" / "windsurf",
            "config": HOME / ".codeium" / "windsurf" / "mcp_config.json",
        },
        # ----- Zed (uses "context_servers" key) ----------------------------
        {
            "id": "zed",
            "name": "Zed",
            "kind": "json-key",
            "key": "context_servers",
            "detect": _xdg_config("zed"),
            "config": _xdg_config("zed") / "settings.json",
        },
        # ----- Continue (VS Code) ------------------------------------------
        {
            "id": "continue",
            "name": "Continue (VS Code)",
            "kind": "json-mcp",
            "detect": HOME / ".continue",
            "config": HOME / ".continue" / "config.json",
        },
        # ----- VS Code MCP (1.99+ native) ----------------------------------
        # VS Code stores MCP config in user settings under "mcp.servers".
        {
            "id": "vscode",
            "name": "VS Code (Copilot)",
            "kind": "json-key",
            "key": "mcp.servers",
            "detect": _platform_app_dir("Code/User", "Code/User", "Code/User"),
            "config": _platform_app_dir("Code/User", "Code/User", "Code/User") / "settings.json",
        },
        # ----- Goose (Block) -----------------------------------------------
        # Goose uses YAML; we don't write it automatically — print instructions.
        {
            "id": "goose",
            "name": "Goose",
            "kind": "yaml-instructions",
            "detect": _xdg_config("goose"),
            "config": _xdg_config("goose") / "config.yaml",
        },
        # ----- OpenClaw (CLI agent) ----------------------------------------
        # Config lives at ~/.openclaw/openclaw.json with nested "mcp.servers".
        {
            "id": "openclaw",
            "name": "OpenClaw",
            "kind": "json-key",
            "key": "mcp.servers",
            "detect": HOME / ".openclaw",
            "config": HOME / ".openclaw" / "openclaw.json",
        },
    ]


def detect_clients() -> list[dict]:
    """Return only the specs whose detect path exists."""
    return [c for c in client_specs() if c["detect"].exists()]


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def backup_file(path: Path) -> Optional[Path]:
    """Back up a file with a timestamp suffix. Returns the backup path, or None if no file existed."""
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, backup)
    return backup


def _set_nested(obj: dict, dotted_key: str, value) -> None:
    """Set obj[a][b][c] = value for dotted_key='a.b.c', creating intermediate dicts."""
    parts = dotted_key.split(".")
    cur = obj
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
        if not isinstance(cur, dict):
            raise ValueError(f"Conflicting non-dict value at '{p}' in config")
    cur[parts[-1]] = value


def _get_nested(obj: dict, dotted_key: str):
    parts = dotted_key.split(".")
    cur = obj
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def write_json_mcp(config_path: Path, key: str, server_name: str, command: str) -> tuple[bool, str]:
    """Write an MCP server entry into a JSON config under `key` (supports dotted keys like 'mcp.servers')."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text() or "{}")
        except json.JSONDecodeError as exc:
            return False, f"existing config is not valid JSON: {exc}"
    else:
        data = {}

    existing_block = _get_nested(data, key)
    if not isinstance(existing_block, dict):
        existing_block = {}

    if server_name in existing_block:
        if existing_block[server_name].get("command") == command:
            return False, "already configured"
        backup_file(config_path)
        existing_block[server_name] = {"command": command}
        _set_nested(data, key, existing_block)
        config_path.write_text(json.dumps(data, indent=2))
        return True, "updated existing entry"

    backup_file(config_path)
    existing_block[server_name] = {"command": command}
    _set_nested(data, key, existing_block)
    config_path.write_text(json.dumps(data, indent=2))
    return True, "added new entry"


def write_toml_mcp(config_path: Path, server_name: str, command: str) -> tuple[bool, str]:
    """Append an [mcp_servers.<name>] section to a TOML config (Codex CLI style).

    Idempotent: if the same section already exists with the same command, no
    change. Otherwise, replace any prior tuiml block.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    section_header = f"[mcp_servers.{server_name}]"
    new_block = f'{section_header}\ncommand = "{command}"\n'

    if not config_path.exists():
        config_path.write_text(new_block)
        return True, "created new config"

    text = config_path.read_text()

    # If the exact line is already present, do nothing.
    if section_header in text and f'command = "{command}"' in text:
        return False, "already configured"

    backup_file(config_path)

    # Remove any existing [mcp_servers.<name>] block (and following lines until next [section] or EOF)
    pattern = re.compile(
        rf"\[mcp_servers\.{re.escape(server_name)}\][^\[]*",
        re.MULTILINE,
    )
    cleaned = pattern.sub("", text).rstrip() + "\n"

    config_path.write_text(cleaned + "\n" + new_block)
    return True, "added entry"


def install_skill_file(skills_dir: Path) -> tuple[bool, str]:
    """Copy the bundled SKILL.md into a coding agent's skills directory."""
    try:
        from importlib.resources import files
        skill_src = files("tuiml.llm").joinpath("Skill.md")
        if not skill_src.is_file():
            return False, f"skill file not found in package: {skill_src}"
        src_text = skill_src.read_text(encoding="utf-8")
    except Exception as exc:
        return False, f"could not read package skill file: {exc}"

    target_dir = skills_dir / "tuiml"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "SKILL.md"

    if target.exists():
        if target.read_text(encoding="utf-8") == src_text:
            return False, "already up to date"
        backup_file(target)

    target.write_text(src_text, encoding="utf-8")
    return True, f"installed to {target}"


def print_yaml_instructions(spec: dict) -> tuple[bool, str]:
    """Print manual setup instructions for YAML-config clients (Goose)."""
    snippet = (
        "extensions:\n"
        "  tuiml:\n"
        "    type: stdio\n"
        "    command: tuiml-mcp\n"
        "    enabled: true\n"
    )
    info(f"  Add to {spec['config']}:")
    for line in snippet.splitlines():
        click.echo(f"      {line}")
    return False, "manual step (YAML config not auto-edited)"


# ---------------------------------------------------------------------------
# Per-client dispatcher
# ---------------------------------------------------------------------------

def configure(spec: dict) -> tuple[bool, str]:
    kind = spec["kind"]
    if kind == "json-mcp":
        return write_json_mcp(spec["config"], "mcpServers", "tuiml", "tuiml-mcp")
    if kind == "json-key":
        return write_json_mcp(spec["config"], spec["key"], "tuiml", "tuiml-mcp")
    if kind == "toml-mcp":
        return write_toml_mcp(spec["config"], "tuiml", "tuiml-mcp")
    if kind == "skill":
        return install_skill_file(spec["skills_dir"])
    if kind == "yaml-instructions":
        return print_yaml_instructions(spec)
    return False, f"unknown client kind: {kind}"


def describe_target(spec: dict) -> str:
    if spec["kind"] == "skill":
        return f"skills dir: {spec['skills_dir']}"
    return str(spec["config"])


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------

ALL_CLIENT_IDS = [c["id"] for c in client_specs()]


@click.command("setup")
@click.option("--yes", "-y", "assume_yes", is_flag=True,
              help="Skip prompts and configure all detected clients.")
@click.option("--list", "list_only", is_flag=True,
              help="List detected clients without making any changes.")
@click.option("--client", "clients", multiple=True,
              help="Configure only the specified client(s). Repeatable. "
                   "Run 'tuiml setup --list' to see valid IDs.")
def setup(assume_yes: bool, list_only: bool, clients: tuple[str, ...]) -> None:
    """Connect TuiML to your AI agents.

    Detects installed clients and wires them up: appends an MCP server entry
    for MCP clients, copies the SKILL.md file for skill-based agents, or
    prints manual instructions for YAML/unsupported configs.

    Supported clients: Claude Desktop, Claude Code, ChatGPT Desktop,
    Perplexity Desktop, Cursor, Windsurf, OpenAI Codex CLI, Zed,
    Continue (VS Code), VS Code (Copilot), Goose, OpenClaw.
    """
    banner()

    info("Detecting installed AI clients ...")
    detected = detect_clients()

    if not detected:
        warn("No AI clients detected.")
        click.echo()
        info("Looked in:")
        for spec in client_specs():
            info(f"  {spec['name']:22} {spec['detect']}")
        click.echo()
        info("Install one of these clients, then re-run: tuiml setup")
        return

    section("Detected:")
    for spec in detected:
        info(f"  {C.GREEN}✓{C.RESET} {spec['name']:22} ({describe_target(spec)})")

    if list_only:
        click.echo()
        return

    # Filter by --client if provided
    if clients:
        unknown = [c for c in clients if c not in ALL_CLIENT_IDS]
        if unknown:
            error(f"Unknown client(s): {', '.join(unknown)}")
            info(f"Valid IDs: {', '.join(ALL_CLIENT_IDS)}")
            sys.exit(1)
        detected = [s for s in detected if s["id"] in clients]
        if not detected:
            error("None of the specified clients were detected on this machine.")
            sys.exit(1)

    section("Configuration:")
    changes_made = 0
    for spec in detected:
        if not confirm(f"Configure {spec['name']}?", default=True, assume_yes=assume_yes):
            info(f"  Skipped {spec['name']}")
            continue

        try:
            changed, reason = configure(spec)
        except Exception as exc:
            error(f"  {spec['name']}: {exc}")
            continue

        if changed:
            success(f"  {spec['name']}: {reason}")
            changes_made += 1
        else:
            info(f"  {spec['name']}: {reason}")

    section("Done.")
    if changes_made:
        info("Restart your AI client(s) to activate TuiML.")
        click.echo()
        info(f"Try asking your agent: {C.CYAN}\"Train a random forest on iris and report accuracy\"{C.RESET}")
    else:
        info("No changes were made.")
    click.echo()
