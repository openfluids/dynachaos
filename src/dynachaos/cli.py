"""Command-line interface for dynachaos pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

from dynachaos import __version__
from dynachaos.pipelines.registry import get_section, list_sections
from dynachaos.pipelines.runner import run_all, run_section


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dynachaos", description="Run dynachaos paper pipelines")
    parser.add_argument("--version", action="version", version=f"dynachaos {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available section IDs")

    run = sub.add_parser("run", help="Run one section or all sections")
    run.add_argument(
        "target",
        help="Section ID (for example sec02_circle_map) or all",
    )
    run.add_argument(
        "--profile",
        choices=("paper", "smoke"),
        default="paper",
        help="paper: allow compute; smoke: require precomputed caches",
    )
    run.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Base output directory (defaults to ./figures)",
    )
    run.add_argument(
        "--recompute",
        action="store_true",
        help="Delete existing outputs for target sections before running",
    )

    style = sub.add_parser("style", help="Inspect and render plotting style themes")
    style_sub = style.add_subparsers(dest="style_command", required=True)
    style_sub.add_parser("list", help="List available style themes")

    preview = style_sub.add_parser("preview", help="Render style preview image(s)")
    preview.add_argument(
        "--theme",
        default=None,
        help="Render only one theme. Omit to render all themes.",
    )
    preview.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/style/themes"),
        help="Directory for generated preview PNG files.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        for section_id in list_sections():
            spec = get_section(section_id)
            modules = ",".join(spec.modules)
            print(f"{section_id}\t{modules}")
        return 0

    if args.command == "style":
        from dynachaos.utils.style import available_themes, save_theme_previews, theme_description

        if args.style_command == "list":
            for theme in available_themes():
                print(f"{theme}\t{theme_description(theme)}")
            return 0

        if args.theme is not None:
            valid = available_themes()
            if args.theme not in valid:
                parser.error(f"invalid choice: '{args.theme}' (choose from {', '.join(valid)})")
        selected = None if args.theme is None else [args.theme]
        for path in save_theme_previews(args.output_dir, themes=selected):
            print(path)
        return 0

    target = args.target

    if target == "all":
        results = run_all(
            output_root=args.output_root,
            profile=args.profile,
            recompute=args.recompute,
        )
        for section_id, paths in results.items():
            print(f"[{section_id}] {len(paths)} outputs")
        return 0

    if target not in set(list_sections()):
        parser.error(f"Unknown target '{target}'. Use 'dynachaos list' for valid sections.")

    paths = run_section(
        target,
        output_root=args.output_root,
        profile=args.profile,
        recompute=args.recompute,
    )
    print(f"[{target}] {len(paths)} outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
