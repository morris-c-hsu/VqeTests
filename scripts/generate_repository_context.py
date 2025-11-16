#!/usr/bin/env python3
"""
Generate a comprehensive repository context PDF for LLM consumption.

Creates a single PDF containing all source code and documentation from the
repository, formatted with clear file delimiters for easy parsing.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib.enums import TA_LEFT


# Patterns to exclude from the context file
EXCLUDE_PATTERNS = {
    # Version control
    '.git', '.gitignore', '.gitattributes',
    # Python
    '__pycache__', '*.pyc', '*.pyo', '*.pyd', '.pytest_cache', '*.egg-info',
    '.venv', 'venv', 'env',
    # IDE
    '.vscode', '.idea', '*.swp', '*.swo', '*~',
    # Build artifacts
    'build', 'dist', '*.so', '*.dylib', '*.dll',
    # Images (binary files)
    '*.png', '*.jpg', '*.jpeg', '*.gif', '*.ico', '*.svg',
    # Other binary/generated files
    '*.pdf', '*.pkl', '*.npy', '*.npz',
    # Logs
    '*.log',
    # OS files
    '.DS_Store', 'Thumbs.db',
}

# File extensions to include (if None, include all text files)
INCLUDE_EXTENSIONS = {
    '.py', '.md', '.txt', '.yml', '.yaml', '.json', '.toml',
    '.sh', '.bash', '.cfg', '.ini', '.conf',
    '.rst', '.tex',
}


def should_exclude(path, exclude_patterns):
    """Check if a path should be excluded based on patterns."""
    path_str = str(path)
    name = path.name

    for pattern in exclude_patterns:
        if pattern.startswith('*'):
            # Wildcard pattern
            if name.endswith(pattern[1:]):
                return True
        elif pattern in path_str or name == pattern:
            return True

    return False


def should_include(path, include_extensions):
    """Check if a file should be included based on extension."""
    if include_extensions is None:
        return True
    return path.suffix in include_extensions


def is_text_file(filepath, max_check_bytes=8192):
    """Check if a file is likely a text file."""
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(max_check_bytes)
            if b'\x00' in chunk:  # Null byte indicates binary
                return False
        return True
    except Exception:
        return False


def collect_files(root_dir, exclude_patterns, include_extensions):
    """Collect all relevant files from the repository."""
    root_path = Path(root_dir).resolve()
    files = []

    for path in root_path.rglob('*'):
        if path.is_file():
            # Check exclusions
            if should_exclude(path, exclude_patterns):
                continue

            # Check inclusions
            if not should_include(path, include_extensions):
                continue

            # Verify it's a text file
            if not is_text_file(path):
                continue

            # Store relative path
            rel_path = path.relative_to(root_path)
            files.append((rel_path, path))

    # Sort files for consistent ordering
    files.sort(key=lambda x: str(x[0]))
    return files


def add_file_to_pdf(story, rel_path, abs_path, styles):
    """Add a file's content to the PDF story."""
    try:
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Add file header
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"<b>FILE: {rel_path}</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))

        # Add content in preformatted style
        # Split into chunks to avoid reportlab issues with very long strings
        max_chunk = 50000
        if len(content) > max_chunk:
            for i in range(0, len(content), max_chunk):
                chunk = content[i:i+max_chunk]
                story.append(Preformatted(chunk, styles['CodeSmall']))
        else:
            story.append(Preformatted(content, styles['CodeSmall']))

        story.append(PageBreak())

    except Exception as e:
        story.append(Paragraph(f"<b>FILE: {rel_path}</b>", styles['Heading2']))
        story.append(Paragraph(f"ERROR: Could not read file: {e}", styles['Normal']))
        story.append(PageBreak())


def generate_repository_context(
    root_dir='.',
    output_file='REPOSITORY_FULL_CONTEXT.pdf',
    exclude_patterns=None,
    include_extensions=None
):
    """Generate the repository context PDF."""
    if exclude_patterns is None:
        exclude_patterns = EXCLUDE_PATTERNS
    if include_extensions is None:
        include_extensions = INCLUDE_EXTENSIONS

    root_path = Path(root_dir).resolve()
    output_path = root_path / 'docs' / output_file

    print("=" * 80)
    print("GENERATING REPOSITORY CONTEXT PDF")
    print("=" * 80)
    print(f"Root directory: {root_path}")
    print(f"Output file: {output_path}")
    print()

    # Collect files
    print("Collecting files...")
    files = collect_files(root_path, exclude_patterns, include_extensions)
    print(f"Found {len(files)} files to include")
    print()

    # Generate PDF
    print("Generating PDF...")

    # Create PDF document
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []

    # Create styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CodeSmall',
        parent=styles['Code'],
        fontSize=7,
        leading=9,
        leftIndent=0,
        rightIndent=0,
        fontName='Courier'
    ))

    # Title page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("<b>REPOSITORY FULL CONTEXT</b>", styles['Title']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"Repository: {root_path.name}", styles['Normal']))
    story.append(Paragraph(f"Total files: {len(files)}", styles['Normal']))
    story.append(PageBreak())

    # Table of contents
    story.append(Paragraph("<b>TABLE OF CONTENTS</b>", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))
    for i, (rel_path, _) in enumerate(files, 1):
        story.append(Paragraph(f"{i:3d}. {rel_path}", styles['Normal']))
    story.append(PageBreak())

    # Add all files
    for i, (rel_path, abs_path) in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {rel_path}")
        add_file_to_pdf(story, rel_path, abs_path, styles)

    # Build PDF
    doc.build(story)

    # Print summary
    file_size = output_path.stat().st_size
    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"Files included: {len(files)}")
    print()
    print("You can now use this PDF for full repository context!")
    print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate repository context PDF for LLM consumption'
    )
    parser.add_argument(
        '--root',
        default='..',
        help='Root directory of repository (default: parent directory)'
    )
    parser.add_argument(
        '--output',
        default='REPOSITORY_FULL_CONTEXT.pdf',
        help='Output filename (default: REPOSITORY_FULL_CONTEXT.pdf)'
    )
    parser.add_argument(
        '--include-images',
        action='store_true',
        help='Include image files (not recommended)'
    )
    parser.add_argument(
        '--all-files',
        action='store_true',
        help='Include all text file types, not just source code'
    )

    args = parser.parse_args()

    # Adjust patterns based on arguments
    exclude_patterns = EXCLUDE_PATTERNS.copy()
    if args.include_images:
        exclude_patterns -= {'*.png', '*.jpg', '*.jpeg', '*.gif', '*.ico', '*.svg'}

    include_extensions = None if args.all_files else INCLUDE_EXTENSIONS

    generate_repository_context(
        root_dir=args.root,
        output_file=args.output,
        exclude_patterns=exclude_patterns,
        include_extensions=include_extensions
    )


if __name__ == '__main__':
    main()
