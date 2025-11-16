#!/usr/bin/env python3
"""
Generate a minimal repository context PDF for LLM consumption.

Includes only:
- All Python source files from src/
- README.md
- docs/IMPLEMENTATION_SUMMARY.md
"""

import os
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted


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


def generate_minimal_context(root_dir='..', output_file='REPOSITORY_MINIMAL_CONTEXT.pdf'):
    """Generate minimal context PDF with src/ code and key docs."""
    root_path = Path(root_dir).resolve()
    output_path = root_path / 'docs' / output_file

    print("=" * 80)
    print("GENERATING MINIMAL REPOSITORY CONTEXT PDF")
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
    story.append(Paragraph("<b>MINIMAL REPOSITORY CONTEXT</b>", styles['Title']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"Repository: {root_path.name}", styles['Normal']))
    story.append(Paragraph(f"Total files: {len(files_to_include)}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Includes:", styles['Normal']))
    story.append(Paragraph("  - README.md", styles['Normal']))
    story.append(Paragraph("  - docs/IMPLEMENTATION_SUMMARY.md", styles['Normal']))
    story.append(Paragraph("  - All Python source files from src/", styles['Normal']))
    story.append(PageBreak())

    # Table of contents
    story.append(Paragraph("<b>TABLE OF CONTENTS</b>", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))
    for i, (rel_path, _) in enumerate(files_to_include, 1):
        story.append(Paragraph(f"{i:2d}. {rel_path}", styles['Normal']))
    story.append(PageBreak())

    # Add all files
    for i, (rel_path, abs_path) in enumerate(files_to_include, 1):
        print(f"  [{i}/{len(files_to_include)}] {rel_path}")
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
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    print(f"Files included: {len(files_to_include)}")
    print()
    print("Minimal context PDF ready for LLM!")
    print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate minimal repository context PDF (src/ + README + IMPLEMENTATION_SUMMARY)'
    )
    parser.add_argument(
        '--root',
        default='..',
        help='Root directory of repository (default: parent directory)'
    )
    parser.add_argument(
        '--output',
        default='REPOSITORY_MINIMAL_CONTEXT.pdf',
        help='Output filename (default: REPOSITORY_MINIMAL_CONTEXT.pdf)'
    )

    args = parser.parse_args()

    generate_minimal_context(
        root_dir=args.root,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
