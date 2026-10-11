#!/usr/bin/env python3
"""Convert reStructuredText files to Markdown format.

This script handles conversion of various RST syntax patterns to their
Markdown equivalents, including headers, code blocks, links, images,
lists, and tables.
"""

from __future__ import annotations

import re
from pathlib import Path


class RSTToMarkdownConverter:
    """Convert reStructuredText to Markdown."""

    def __init__(self, content: str) -> None:
        """Initialize converter with RST content.

        Args:
            content: The RST content as a string

        """
        self.content = content
        self.lines = content.split("\n")
        self.converted_lines: list[str] = []

    def convert(self) -> str:
        """Convert RST content to Markdown.

        Returns:
            Converted Markdown content

        """
        i = 0
        while i < len(self.lines):
            line = self.lines[i]

            # Check for section headers (underlined with =, -, ~, ^, etc.)
            if i + 1 < len(self.lines):
                next_line = self.lines[i + 1]
                header_result = self._convert_header(line, next_line)
                if header_result:
                    self.converted_lines.append(header_result)
                    i += 2  # Skip the underline
                    continue

            # Check for code blocks
            if line.strip().startswith(".. code::"):
                code_block, lines_consumed = self._convert_code_block(i)
                self.converted_lines.extend(code_block)
                i += lines_consumed
                continue

            # Check for images
            if line.strip().startswith(".. image::"):
                image, lines_consumed = self._convert_image(i)
                self.converted_lines.extend(image)
                i += lines_consumed
                continue

            # Check for raw HTML blocks
            if line.strip().startswith(".. raw::"):
                raw_block, lines_consumed = self._convert_raw_block(i)
                self.converted_lines.extend(raw_block)
                i += lines_consumed
                continue

            # Check for parsed-literal blocks
            if line.strip().startswith(".. parsed-literal::"):
                literal_block, lines_consumed = self._convert_literal_block(i)
                self.converted_lines.extend(literal_block)
                i += lines_consumed
                continue

            # Check for figure directive
            if line.strip().startswith(".. figure::"):
                figure, lines_consumed = self._convert_figure(i)
                self.converted_lines.extend(figure)
                i += lines_consumed
                continue

            # Check for toctree directive
            if line.strip().startswith(".. toctree::"):
                toctree, lines_consumed = self._convert_toctree(i)
                self.converted_lines.extend(toctree)
                i += lines_consumed
                continue

            # Check for topic directive
            if line.strip().startswith(".. topic::"):
                topic, lines_consumed = self._convert_topic(i)
                self.converted_lines.extend(topic)
                i += lines_consumed
                continue

            # Check for autodoc directives (Sphinx-specific)
            if line.strip().startswith(".. auto"):
                autodoc, lines_consumed = self._convert_autodoc(i)
                self.converted_lines.extend(autodoc)
                i += lines_consumed
                continue

            # Skip RST comments
            if line.strip().startswith(".. ") and not any(
                line.strip().startswith(d)
                for d in [
                    ".. code::",
                    ".. image::",
                    ".. figure::",
                    ".. raw::",
                    ".. parsed-literal::",
                    ".. toctree::",
                    ".. topic::",
                    ".. auto",
                ]
            ):
                # This is likely a comment or unsupported directive, preserve as HTML comment
                self.converted_lines.append(f"<!-- {line.strip()} -->")
                i += 1
                continue

            # Convert inline markup and links
            converted_line = self._convert_inline_markup(line)
            self.converted_lines.append(converted_line)
            i += 1

        return "\n".join(self.converted_lines)

    def _convert_header(self, line: str, next_line: str) -> str | None:
        """Convert RST header to Markdown.

        Args:
            line: The header text line
            next_line: The underline line

        Returns:
            Markdown header or None if not a header

        """
        line = line.strip()
        next_line = next_line.strip()

        if not line or not next_line:
            return None

        # Check if next_line is all the same character
        if len(set(next_line)) == 1 and len(next_line) >= len(line):
            char = next_line[0]
            # Determine header level based on character
            level_map = {
                "=": 1,
                "-": 2,
                "~": 3,
                "^": 4,
                '"': 5,
                "'": 6,
            }
            level = level_map.get(char, 2)
            return f"{'#' * level} {line}\n"

        return None

    def _convert_code_block(self, start_idx: int) -> tuple[list[str], int]:
        """Convert RST code block to Markdown.

        Args:
            start_idx: Starting line index

        Returns:
            Tuple of (converted lines, number of lines consumed)

        """
        line = self.lines[start_idx].strip()
        # Extract language if specified
        match = re.match(r"\.\.\s+code::\s*(.+)?", line)
        language = match.group(1).strip() if match and match.group(1) else ""

        result = [f"```{language}"]
        i = start_idx + 1

        # Skip empty line after directive
        if i < len(self.lines) and not self.lines[i].strip():
            i += 1

        # Collect indented code lines
        while i < len(self.lines):
            line = self.lines[i]
            # Code blocks are indented; stop when we hit a non-indented line
            if (
                line
                and not line.startswith("    ")
                and not line.startswith("\t")
                and line.strip()
            ):
                break
            # Add the line, removing the indent
            if line.strip():
                result.append(
                    line[4:] if line.startswith("    ") else line.removeprefix("\t")
                )
            else:
                result.append("")
            i += 1

        result.append("```")
        result.append("")

        return result, i - start_idx

    def _convert_image(self, start_idx: int) -> tuple[list[str], int]:
        """Convert RST image directive to Markdown.

        Args:
            start_idx: Starting line index

        Returns:
            Tuple of (converted lines, number of lines consumed)

        """
        line = self.lines[start_idx].strip()
        match = re.match(r"\.\.\s+image::\s*(.+)", line)
        if not match:
            return [line], 1

        image_path = match.group(1).strip()
        alt_text = ""
        width = ""

        i = start_idx + 1
        # Parse image options
        while i < len(self.lines):
            line = self.lines[i]
            if line and not line.startswith("   ") and not line.startswith("\t"):
                break

            line = line.strip()
            if line.startswith(":alt:"):
                alt_text = line.replace(":alt:", "").strip()
            elif line.startswith(":width:"):
                width = line.replace(":width:", "").strip()

            i += 1

        # Create Markdown image syntax
        if not alt_text:
            alt_text = "Image"

        result = [f"![{alt_text}]({image_path})"]
        if width:
            result.append(f"<!-- width: {width} -->")
        result.append("")

        return result, i - start_idx

    def _convert_figure(self, start_idx: int) -> tuple[list[str], int]:
        """Convert RST figure directive to Markdown.

        Args:
            start_idx: Starting line index

        Returns:
            Tuple of (converted lines, number of lines consumed)

        """
        line = self.lines[start_idx].strip()
        match = re.match(r"\.\.\s+figure::\s*(.+)", line)
        if not match:
            return [line], 1

        image_path = match.group(1).strip()
        alt_text = ""
        caption = ""

        i = start_idx + 1
        # Parse figure options and caption
        while i < len(self.lines):
            line = self.lines[i]
            if line and not line.startswith("   ") and not line.startswith("\t"):
                break

            stripped = line.strip()
            if stripped.startswith(":alt:"):
                alt_text = stripped.replace(":alt:", "").strip()
            elif stripped and not stripped.startswith(":"):
                # This is the caption
                caption = stripped

            i += 1

        if not alt_text:
            alt_text = caption or "Figure"

        result = [f"![{alt_text}]({image_path})"]
        if caption:
            result.append(f"*{caption}*")
        result.append("")

        return result, i - start_idx

    def _convert_raw_block(self, start_idx: int) -> tuple[list[str], int]:
        """Convert RST raw directive to Markdown (preserve as-is or skip).

        Args:
            start_idx: Starting line index

        Returns:
            Tuple of (converted lines, number of lines consumed)

        """
        line = self.lines[start_idx].strip()
        match = re.match(r"\.\.\s+raw::\s*(.+)", line)
        if not match:
            return [line], 1

        format_type = match.group(1).strip()
        result = []
        i = start_idx + 1

        # Check if it's a file include
        if i < len(self.lines) and ":file:" in self.lines[i]:
            file_match = re.search(r":file:\s*(.+)", self.lines[i])
            if file_match:
                file_path = file_match.group(1).strip()
                result.append(f"[View {format_type} file]({file_path})")
                result.append("")
            i += 1
            return result, i - start_idx

        # Skip empty line after directive
        if i < len(self.lines) and not self.lines[i].strip():
            i += 1

        # If it's HTML, we can preserve it
        if format_type.lower() == "html":
            result.append("")
            # Collect the raw HTML content
            while i < len(self.lines):
                line = self.lines[i]
                # Raw blocks are indented; stop when we hit a non-indented line
                if (
                    line
                    and not line.startswith("    ")
                    and not line.startswith("\t")
                    and line.strip()
                ):
                    break
                # Add the line, removing the indent
                if line.strip():
                    content = (
                        line[4:] if line.startswith("    ") else line.removeprefix("\t")
                    )
                    result.append(content)
                else:
                    result.append("")
                i += 1
            result.append("")
        else:
            # Skip other raw formats
            while i < len(self.lines):
                line = self.lines[i]
                if (
                    line
                    and not line.startswith("    ")
                    and not line.startswith("\t")
                    and line.strip()
                ):
                    break
                i += 1

        return result, i - start_idx

    def _convert_literal_block(self, start_idx: int) -> tuple[list[str], int]:
        """Convert RST parsed-literal directive to Markdown code block.

        Args:
            start_idx: Starting line index

        Returns:
            Tuple of (converted lines, number of lines consumed)

        """
        result = ["```"]
        i = start_idx + 1

        # Skip empty line after directive
        if i < len(self.lines) and not self.lines[i].strip():
            i += 1

        # Collect indented literal lines
        while i < len(self.lines):
            line = self.lines[i]
            if (
                line
                and not line.startswith("    ")
                and not line.startswith("\t")
                and line.strip()
            ):
                break
            if line.strip():
                result.append(
                    line[4:] if line.startswith("    ") else line.removeprefix("\t")
                )
            else:
                result.append("")
            i += 1

        result.append("```")
        result.append("")

        return result, i - start_idx

    def _convert_toctree(self, start_idx: int) -> tuple[list[str], int]:
        """Convert RST toctree directive to Markdown list.

        Args:
            start_idx: Starting line index

        Returns:
            Tuple of (converted lines, number of lines consumed)

        """
        line = self.lines[start_idx].strip()
        i = start_idx + 1

        # Parse options
        caption = ""
        while i < len(self.lines):
            line = self.lines[i]
            if not line.strip():
                i += 1
                continue
            if line.strip().startswith(":caption:"):
                caption = line.strip().replace(":caption:", "").strip()
                i += 1
                continue
            if line.strip().startswith(":"):
                i += 1
                continue
            break

        result = []
        if caption:
            result.append(f"## {caption}")
            result.append("")

        # Collect toctree entries
        while i < len(self.lines):
            line = self.lines[i]
            if (
                line
                and not line.startswith("   ")
                and not line.startswith("\t")
                and line.strip()
            ):
                break

            entry = line.strip()
            if entry:
                # Convert to list item with link
                result.append(f"- [{entry}]({entry})")

            i += 1

        result.append("")
        return result, i - start_idx

    def _convert_topic(self, start_idx: int) -> tuple[list[str], int]:
        """Convert RST topic directive to Markdown blockquote.

        Args:
            start_idx: Starting line index

        Returns:
            Tuple of (converted lines, number of lines consumed)

        """
        line = self.lines[start_idx].strip()
        match = re.match(r"\.\.\s+topic::\s*(.+)?", line)
        title = match.group(1).strip() if match and match.group(1) else ""

        result = []
        if title:
            result.append(f"**{title}**")
            result.append("")

        i = start_idx + 1
        # Skip empty line
        if i < len(self.lines) and not self.lines[i].strip():
            i += 1

        # Collect topic content as blockquote
        while i < len(self.lines):
            line = self.lines[i]
            if (
                line
                and not line.startswith("   ")
                and not line.startswith("\t")
                and not line.startswith("  ")
            ):
                break

            content = line.strip()
            if content:
                result.append(f"> {content}")
            else:
                result.append(">")

            i += 1

        result.append("")
        return result, i - start_idx

    def _convert_autodoc(self, start_idx: int) -> tuple[list[str], int]:
        """Convert Sphinx autodoc directive to Markdown note.

        Args:
            start_idx: Starting line index

        Returns:
            Tuple of (converted lines, number of lines consumed)

        """
        line = self.lines[start_idx].strip()

        # Extract directive type and target
        match = re.match(
            r"\.\.\s+(autoclass|automodule|autofunction|automethod)::\s*(.+)", line
        )
        if not match:
            return [f"<!-- {line} -->"], 1

        directive_type = match.group(1)
        target = match.group(2).strip()

        result = []
        if directive_type == "autoclass":
            result.append(f"## {target}")
            result.append("")
            result.append(f"> **API Reference:** `{target}`")
            result.append(">")
            result.append(
                "> This is an auto-generated API reference. See the Python docstrings for details."
            )
        elif directive_type == "automodule":
            result.append(f"## Module: {target}")
            result.append("")
            result.append(f"> **API Reference:** Module `{target}`")
            result.append(">")
            result.append(
                "> This is an auto-generated API reference. See the Python docstrings for details."
            )
        else:
            result.append(f"### {target}")
            result.append("")
            result.append(f"> **API Reference:** `{target}`")

        result.append("")

        i = start_idx + 1
        # Skip options (lines starting with :)
        while i < len(self.lines):
            line = self.lines[i]
            if not line.strip() or (line.strip() and not line.strip().startswith(":")):
                break
            i += 1

        return result, i - start_idx

    def _convert_inline_markup(self, line: str) -> str:
        """Convert inline RST markup to Markdown.

        Args:
            line: Line with potential inline markup

        Returns:
            Line with Markdown markup

        """
        # Convert reference links: `text <url>`_
        line = re.sub(r"`([^<>`]+)\s+<([^>]+)>`_+", r"[\1](\2)", line)

        # Convert simple external links: `text <url>`__
        line = re.sub(r"`([^<>`]+)\s+<([^>]+)>`__", r"[\1](\2)", line)

        # Convert role-based links like :meth:`~class.method`
        line = re.sub(r":meth:`~?([^`]+)`", r"`\1`", line)
        line = re.sub(r":class:`~?([^`]+)`", r"`\1`", line)
        line = re.sub(r":func:`~?([^`]+)`", r"`\1`", line)
        line = re.sub(r":ref:`([^`]+)`", r"`\1`", line)

        # Convert strong emphasis: **text** (already compatible)
        # Convert emphasis: *text* (already compatible)

        # Convert inline code: ``text`` to `text`
        return re.sub(r"``([^`]+)``", r"`\1`", line)


def convert_file(input_path: Path, output_path: Path | None = None) -> bool:
    """Convert a single RST file to Markdown.

    Args:
        input_path: Path to input RST file
        output_path: Path to output MD file (defaults to same name with .md extension)

    Returns:
        True if conversion successful, False otherwise

    """
    try:
        content = input_path.read_text(encoding="utf-8")

        converter = RSTToMarkdownConverter(content)
        markdown_content = converter.convert()

        if output_path is None:
            output_path = input_path.with_suffix(".md")

        output_path.write_text(markdown_content, encoding="utf-8")
        return True
    except Exception:
        return False


def main() -> None:
    """Main function to convert all RST files."""
    # Define the RST files to convert
    rst_files = [
        "/home/georgepearse/umap/doc/exploratory_analysis.rst",
        "/home/georgepearse/umap/doc/densmap_demo.rst",
        "/home/georgepearse/umap/doc/reproducibility.rst",
        "/home/georgepearse/umap/doc/development_roadmap.rst",
        "/home/georgepearse/umap/doc/index.rst",
        "/home/georgepearse/umap/doc/faq.rst",
        "/home/georgepearse/umap/doc/outliers.rst",
        "/home/georgepearse/umap/doc/aligned_umap_basic_usage.rst",
        "/home/georgepearse/umap/doc/precomputed_k-nn.rst",
        "/home/georgepearse/umap/doc/basic_usage.rst",
        "/home/georgepearse/umap/doc/api.rst",
        "/home/georgepearse/umap/doc/aligned_umap_politics_demo.rst",
        "/home/georgepearse/umap/doc/plotting.rst",
        "/home/georgepearse/umap/doc/parametric_umap.rst",
        "/home/georgepearse/umap/doc/performance.rst",
        "/home/georgepearse/umap/doc/parameters.rst",
        "/home/georgepearse/umap/doc/nomic_atlas_visualizing_mnist_training_dynamics.rst",
        "/home/georgepearse/umap/doc/nomic_atlas_umap_of_text_embeddings.rst",
        "/home/georgepearse/umap/doc/mutual_nn_umap.rst",
        "/home/georgepearse/umap/doc/inverse_transform.rst",
        "/home/georgepearse/umap/doc/interactive_viz.rst",
        "/home/georgepearse/umap/doc/how_umap_works.rst",
        "/home/georgepearse/umap/doc/embedding_space.rst",
        "/home/georgepearse/umap/doc/document_embedding.rst",
        "/home/georgepearse/umap/doc/composing_models.rst",
        "/home/georgepearse/umap/doc/clustering.rst",
        "/home/georgepearse/umap/doc/benchmarking.rst",
        "/home/georgepearse/umap/doc/transform_landmarked_pumap.rst",
        "/home/georgepearse/umap/doc/transform.rst",
        "/home/georgepearse/umap/doc/supervised.rst",
        "/home/georgepearse/umap/doc/sparse.rst",
        "/home/georgepearse/umap/doc/scientific_papers.rst",
        "/home/georgepearse/umap/doc/release_notes.rst",
    ]

    successful = 0
    failed = 0

    for rst_file_path in rst_files:
        path = Path(rst_file_path)
        if not path.exists():
            failed += 1
            continue

        if convert_file(path):
            successful += 1
        else:
            failed += 1

    if successful > 0:
        for rst_file_path in rst_files:
            md_path = Path(rst_file_path).with_suffix(".md")
            if md_path.exists():
                pass


if __name__ == "__main__":
    main()
