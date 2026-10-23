"""Shell completion scripts for bash, zsh and fish, generated from the argument parser.

`mesh completion bash` (or zsh, fish) prints a script to add to your shell once. It completes commands, options,
config files, and, for `mesh logs`, the names of your services (asked of `mesh _complete services`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

SHELLS = ("bash", "zsh", "fish")
DIR_OPTIONS = {"--log-dir"}
FILE_OPTIONS = {"--gpus-csv"}


@dataclass
class Option:
    flags: list[str]
    help: str
    takes_value: bool
    kind: str  # "dir" | "file" | "none"


@dataclass
class Command:
    name: str
    help: str
    options: list[Option] = field(default_factory=list)
    positional: str | None = None  # "config" | "service" | "shell" | None
    choices: list[str] = field(default_factory=list)


def _clean(text: str | None) -> str:
    return (text or "").replace("'", "").replace("`", "").replace('"', "").replace("\n", " ").replace("%", "%%").strip()


def introspect(parser: argparse.ArgumentParser) -> tuple[list[Option], list[Command]]:
    """Top-level options and the visible subcommands (those whose name does not start with an underscore)."""
    top: list[Option] = []
    commands: list[Command] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            helps = {choice.dest: choice.help for choice in action._choices_actions}
            for name, sub in action.choices.items():
                if name.startswith("_"):
                    continue
                command = Command(name, _clean(helps.get(name)))
                for a in sub._actions:
                    if isinstance(a, argparse._HelpAction):
                        continue
                    if a.option_strings:
                        flag = a.option_strings[0]
                        kind = "dir" if flag in DIR_OPTIONS else "file" if flag in FILE_OPTIONS else "none"
                        takes = a.nargs != 0
                        command.options.append(Option(list(a.option_strings), _clean(a.help), takes, kind))
                    else:
                        command.positional = a.dest
                        command.choices = [str(c) for c in (a.choices or [])]
                commands.append(command)
        elif action.option_strings and not isinstance(action, argparse._HelpAction):
            top.append(Option(list(action.option_strings), _clean(action.help), action.nargs != 0, "none"))
    return top, commands


def bash(parser: argparse.ArgumentParser) -> str:
    top, commands = introspect(parser)
    names = " ".join(c.name for c in commands)
    top_flags = " ".join(f for o in top for f in o.flags) + " --help"
    cases = []
    for c in commands:
        flags = " ".join(f for o in c.options for f in o.flags) + " --help"
        cases.append(f'    {c.name}) opts="{flags}" ;;')
    value_dirs = "|".join(sorted(DIR_OPTIONS))
    value_files = "|".join(sorted(FILE_OPTIONS))
    shells = " ".join(SHELLS)
    return f"""# bash completion for mesh. Add to ~/.bashrc:  eval "$(mesh completion bash)"
_mesh() {{
    local cur prev cmd opts logdir i
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    cmd="${{COMP_WORDS[1]}}"
    if [ "$COMP_CWORD" -eq 1 ]; then
        COMPREPLY=($(compgen -W "{names} {top_flags}" -- "$cur"))
        return
    fi
    case "$prev" in
        {value_dirs}) COMPREPLY=($(compgen -d -- "$cur")); return ;;
        {value_files}) COMPREPLY=($(compgen -f -- "$cur")); return ;;
    esac
    if [[ "$cur" == -* ]]; then
        case "$cmd" in
{chr(10).join(cases)}
        esac
        COMPREPLY=($(compgen -W "$opts" -- "$cur"))
        return
    fi
    case "$cmd" in
        plan|up) COMPREPLY=($(compgen -f -- "$cur")) ;;
        logs)
            logdir=logs
            for ((i = 1; i < COMP_CWORD; i++)); do
                if [ "${{COMP_WORDS[i]}}" = "--log-dir" ]; then logdir="${{COMP_WORDS[i+1]}}"; fi
            done
            COMPREPLY=($(compgen -W "$(mesh _complete services --log-dir "$logdir" 2>/dev/null)" -- "$cur"))
            ;;
        completion) COMPREPLY=($(compgen -W "{shells}" -- "$cur")) ;;
    esac
}}
complete -o filenames -F _mesh mesh
"""


def zsh(parser: argparse.ArgumentParser) -> str:
    top, commands = introspect(parser)
    command_lines = "\n".join(f"        '{c.name}:{c.help}'" for c in commands)
    cases = []
    for c in commands:
        specs = []
        for o in c.options:
            where = {"dir": ":directory:_files -/", "file": ":file:_files", "none": ":value: " if o.takes_value else ""}[o.kind]
            for flag in o.flags:
                specs.append(f"'{flag}[{o.help}]{where}'")
        if c.positional == "config":
            specs.append("':config:_files -g \"*.(yaml|yml)\"'")
        elif c.positional == "service":
            specs.append("':service:_mesh_services'")
        elif c.positional == "shell":
            specs.append(f"':shell:({' '.join(c.choices)})'")
        body = " \\\n                ".join(specs) if specs else "'*:'"
        cases.append(f"        {c.name})\n            _arguments \\\n                {body}\n            ;;")
    top_specs = " ".join(f"'{f}[{o.help}]'" for o in top for f in o.flags)
    return f"""#compdef mesh
# zsh completion for mesh. Add to ~/.zshrc (after compinit):  eval "$(mesh completion zsh)"
_mesh_services() {{
    local -a names
    names=(${{(f)"$(mesh _complete services --log-dir "${{opt_args[--log-dir]:-logs}}" 2>/dev/null)"}})
    _describe 'service' names
}}

_mesh() {{
    local -a commands
    commands=(
{command_lines}
    )
    if (( CURRENT == 2 )); then
        _describe 'command' commands
        _arguments {top_specs}
        return
    fi
    case "$words[2]" in
{chr(10).join(cases)}
    esac
}}

compdef _mesh mesh
"""


def fish(parser: argparse.ArgumentParser) -> str:
    top, commands = introspect(parser)
    lines = ["# fish completion for mesh. Save to ~/.config/fish/completions/mesh.fish:  mesh completion fish > that file",
             "complete -c mesh -f"]
    for o in top:
        long = [f[2:] for f in o.flags if f.startswith("--")]
        lines.append(f"complete -c mesh -n '__fish_use_subcommand' -l {long[0]} -d '{o.help}'")
    for c in commands:
        lines.append(f"complete -c mesh -n '__fish_use_subcommand' -a {c.name} -d '{c.help}'")
    for c in commands:
        for o in c.options:
            long = [f[2:] for f in o.flags if f.startswith("--")]
            short = [f[1:] for f in o.flags if not f.startswith("--") and f.startswith("-")]
            parts = [f"complete -c mesh -n '__fish_seen_subcommand_from {c.name}'"]
            if long:
                parts.append(f"-l {long[0]}")
            if short:
                parts.append(f"-s {short[0]}")
            if o.takes_value:
                parts.append("-r -F" if o.kind in ("dir", "file") else "-r")
            parts.append(f"-d '{o.help}'")
            lines.append(" ".join(parts))
        if c.positional == "config":
            lines.append(f"complete -c mesh -n '__fish_seen_subcommand_from {c.name}' -F")
        elif c.positional == "service":
            lines.append(f"complete -c mesh -n '__fish_seen_subcommand_from {c.name}' -a '(mesh _complete services 2>/dev/null)'")
        elif c.positional == "shell":
            lines.append(f"complete -c mesh -n '__fish_seen_subcommand_from {c.name}' -a '{' '.join(c.choices)}'")
    return "\n".join(lines) + "\n"


def script(shell: str, parser: argparse.ArgumentParser) -> str:
    return {"bash": bash, "zsh": zsh, "fish": fish}[shell](parser)
