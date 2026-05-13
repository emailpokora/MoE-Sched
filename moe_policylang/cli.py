"""Command-line interface for MoE-PolicyLang.

Usage:
    moe-policylang validate FILE [FILE ...]   Parse and validate .moe policy files.
    moe-policylang parse FILE                 Parse a .moe file and print the IR.
    moe-policylang version                    Print the MoE-PolicyLang version.

Examples:
    moe-policylang validate examples/lru_policy.moe
    moe-policylang parse examples/composed_policy.moe
    python -m moe_policylang validate examples/*.moe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from moe_policylang import __version__
from moe_policylang.errors import DSLError, ValidationError
from moe_policylang.parser import parse_file


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate one or more .moe files."""
    errors = 0
    for filepath in args.files:
        path = Path(filepath)
        try:
            policies = parse_file(path)
            count = len(policies)
            names = ", ".join(p.name for p in policies)
            print(f"  \u2713 {path}  ({count} {'policy' if count == 1 else 'policies'}: {names})")
        except (DSLError, ValidationError) as e:
            print(f"  \u2717 {path}  {e}", file=sys.stderr)
            errors += 1
        except Exception as e:
            print(f"  \u2717 {path}  unexpected error: {e}", file=sys.stderr)
            errors += 1

    total = len(args.files)
    passed = total - errors
    if errors:
        print(f"\n{passed}/{total} files passed, {errors} failed.", file=sys.stderr)
        return 1
    print(f"\n{passed}/{total} files passed.")
    return 0


def _cmd_parse(args: argparse.Namespace) -> int:
    """Parse a .moe file and print the IR."""
    path = Path(args.file)
    try:
        policies = parse_file(path)
    except (DSLError, ValidationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    for ir in policies:
        print(f"Policy: {ir.name}")
        print(f"  Cache:    capacity={ir.cache.capacity}  eviction={ir.cache.eviction.value}")
        if ir.cache.pin_experts:
            print(f"            pin={ir.cache.pin_experts}")
        print(f"  Prefetch: strategy={ir.prefetch.strategy.value}  budget={ir.prefetch.budget}")
        print(f"  Schedule: mode={ir.schedule.mode.value}  cpu_threshold_ms={ir.schedule.cpu_threshold_ms}")
        if ir.monitor:
            print(f"  Monitor:  metrics={ir.monitor.metrics}  window={ir.monitor.window}")
        if ir.adapt:
            print(f"  Adapt:    {len(ir.adapt.rules)} rule(s)")
            for i, rule in enumerate(ir.adapt.rules):
                c = rule.condition
                a = rule.action
                window = f" for {c.window} accesses" if c.window > 1 else ""
                print(f"            [{i}] when {c.metric} {c.op} {c.threshold}{window} -> {a.param} = {a.value}")
        print()
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    """Print version."""
    print(f"moe-policylang {__version__}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moe-policylang",
        description="MoE-PolicyLang: Domain-specific language for MoE expert scheduling policies.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    # validate
    val_parser = sub.add_parser("validate", help="Parse and validate .moe policy files")
    val_parser.add_argument("files", nargs="+", help="One or more .moe files")
    val_parser.set_defaults(func=_cmd_validate)

    # parse
    parse_parser = sub.add_parser("parse", help="Parse a .moe file and print the IR")
    parse_parser.add_argument("file", help="A .moe file")
    parse_parser.set_defaults(func=_cmd_parse)

    # version
    ver_parser = sub.add_parser("version", help="Print version")
    ver_parser.set_defaults(func=_cmd_version)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
