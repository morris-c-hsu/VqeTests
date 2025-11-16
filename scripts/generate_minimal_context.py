#!/usr/bin/env python3
"""
Generate a minimal repository context file for LLM consumption.

Includes only:
- All Python source files from src/
- README.md
- docs/IMPLEMENTATION_SUMMARY.md
"""

import os
from pathlib import Path
from datetime import datetime


def format_file_content(rel_path, abs_path):
    """Format a file's content with clear delimiters."""
    try:
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        header = f"\n{'=' * 80}\n"
        header += f"FILE: {rel_path}\n"
        header += f"{'=' * 80}\n\n"

        return header + content + "\n"

    except Exception as e:
        return f"\n{'=' * 80}\nFILE: {rel_path}\nERROR: Could not read file: {e}\n{'=' * 80}\n\n"


def generate_minimal_context(root_dir='..', output_file='REPOSITORY_MINIMAL_CONTEXT.txt'):
    """Generate minimal context file with src/ code and key docs."""
    root_path = Path(root_dir).resolve()
    output_path = root_path / 'docs' / output_file

    print("=" * 80)
    print("GENERATING MINIMAL REPOSITORY CONTEXT")
    print("=" * 80)
    print(f"Root directory: {root_path}")
    print(f"Output file: {output_path}")
    print()

    # Define files to include
    files_to_include = []

    # 1. README.md
    readme_path = root_path / 'README.md'
    if readme_path.exists():
        files_to_include.append(('README.md', readme_path))

    # 2. IMPLEMENTATION_SUMMARY.md
    impl_summary_path = root_path / 'docs' / 'IMPLEMENTATION_SUMMARY.md'
    if impl_summary_path.exists():
        files_to_include.append(('docs/IMPLEMENTATION_SUMMARY.md', impl_summary_path))

    # 3. All Python files in src/
    src_dir = root_path / 'src'
    if src_dir.exists():
        for py_file in sorted(src_dir.glob('*.py')):
            rel_path = py_file.relative_to(root_path)
            files_to_include.append((str(rel_path), py_file))

    print(f"Files to include: {len(files_to_include)}")
    for rel_path, _ in files_to_include:
        print(f"  - {rel_path}")
    print()

    # Generate context file
    print("Generating context file...")
    with open(output_path, 'w', encoding='utf-8') as out:
        # Write header
        out.write("=" * 80 + "\n")
        out.write("MINIMAL REPOSITORY CONTEXT\n")
        out.write("=" * 80 + "\n\n")
        out.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"Repository: {root_path.name}\n")
        out.write(f"Total files: {len(files_to_include)}\n\n")

        out.write("Includes:\n")
        out.write("  - README.md\n")
        out.write("  - docs/IMPLEMENTATION_SUMMARY.md\n")
        out.write("  - All Python source files from src/\n\n")

        # Write table of contents
        out.write("=" * 80 + "\n")
        out.write("TABLE OF CONTENTS\n")
        out.write("=" * 80 + "\n\n")
        for i, (rel_path, _) in enumerate(files_to_include, 1):
            out.write(f"{i:2d}. {rel_path}\n")
        out.write("\n")

        # Write file contents
        out.write("=" * 80 + "\n")
        out.write("FILE CONTENTS\n")
        out.write("=" * 80 + "\n")

        for i, (rel_path, abs_path) in enumerate(files_to_include, 1):
            print(f"  [{i}/{len(files_to_include)}] {rel_path}")
            content = format_file_content(rel_path, abs_path)
            out.write(content)

    # Print summary
    file_size = output_path.stat().st_size
    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    print(f"Files included: {len(files_to_include)}")
    print()
    print("Minimal context ready for LLM!")
    print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate minimal repository context (src/ + README + IMPLEMENTATION_SUMMARY)'
    )
    parser.add_argument(
        '--root',
        default='..',
        help='Root directory of repository (default: parent directory)'
    )
    parser.add_argument(
        '--output',
        default='REPOSITORY_MINIMAL_CONTEXT.txt',
        help='Output filename (default: REPOSITORY_MINIMAL_CONTEXT.txt)'
    )

    args = parser.parse_args()

    generate_minimal_context(
        root_dir=args.root,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
