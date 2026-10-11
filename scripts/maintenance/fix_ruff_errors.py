#!/usr/bin/env python3
"""Script to automatically fix ruff errors in the UMAP codebase."""

import ast
import subprocess
from pathlib import Path


def run_ruff_check(file_path: str) -> list[str]:
    """Run ruff check on a file and return the errors."""
    result = subprocess.run(
        ["ruff", "check", file_path, "--output-format", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and result.stdout:
        import json

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return []
    return []


def fix_line_length(line: str, max_length: int = 88) -> list[str]:
    """Fix line length issues by splitting long lines."""
    if len(line) <= max_length:
        return [line]

    # Handle docstrings
    if '"""' in line or "'''" in line:
        # Find indentation
        indent = len(line) - len(line.lstrip())
        # Split at word boundaries
        words = line.strip().split()
        lines = []
        current_line = " " * indent
        for word in words:
            if len(current_line + " " + word) <= max_length:
                if current_line.strip():
                    current_line += " " + word
                else:
                    current_line += word
            else:
                lines.append(current_line)
                current_line = " " * indent + word
        if current_line.strip():
            lines.append(current_line)
        return lines

    # Handle regular code lines
    # Try to split at operators or commas
    if "," in line:
        parts = line.split(",")
        if len(parts) > 1:
            indent = len(line) - len(line.lstrip())
            lines = [parts[0] + ","]
            for i, part in enumerate(parts[1:], 1):
                if i < len(parts) - 1:
                    lines.append(" " * (indent + 4) + part.strip() + ",")
                else:
                    lines.append(" " * (indent + 4) + part.strip())
            return lines

    return [line]


def add_type_annotations(file_path: str) -> str:
    """Add basic type annotations to functions."""
    with open(file_path) as f:
        content = f.read()

    # Parse the AST to understand the code structure
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content

    lines = content.splitlines()

    # Track imports we need to add
    imports_needed = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if function already has return annotation
            if not node.returns and node.name != "__init__":
                # Add basic return annotation based on function name
                if node.name.startswith("is_") or node.name.startswith("has_"):
                    # Boolean returning functions
                    line_num = node.lineno - 1
                    if line_num < len(lines):
                        line = lines[line_num]
                        if "):$" in line:
                            lines[line_num] = line[:-1] + " -> bool:"
                        elif "):" in line:
                            lines[line_num] = line.replace("):", ") -> bool:")
                elif node.name.startswith("get_"):
                    # Getter functions - likely return Any for now
                    imports_needed.add("from typing import Any")
                    line_num = node.lineno - 1
                    if line_num < len(lines):
                        line = lines[line_num]
                        if "):$" in line:
                            lines[line_num] = line[:-1] + " -> Any:"
                        elif "):" in line:
                            lines[line_num] = line.replace("):", ") -> Any:")

    # Add imports at the top if needed
    if imports_needed:
        # Find where to insert imports
        insert_line = 0
        for i, line in enumerate(lines):
            if line.startswith(("import ", "from ")):
                insert_line = i + 1
            elif line and not line.startswith("#") and insert_line > 0:
                break

        for imp in sorted(imports_needed):
            lines.insert(insert_line, imp)
            insert_line += 1

    return "\n".join(lines)


def fix_file(file_path: str) -> None:
    """Fix ruff errors in a single file."""
    # First, try auto-fix with ruff
    subprocess.run(
        ["ruff", "check", file_path, "--fix", "--unsafe-fixes"],
        check=False,
        capture_output=True,
    )

    # Then handle specific error types
    errors = run_ruff_check(file_path)

    if not errors:
        return

    with open(file_path) as f:
        lines = f.readlines()

    # Group errors by type
    error_types = {}
    for error in errors:
        code = error.get("code", "")
        if code not in error_types:
            error_types[code] = []
        error_types[code].append(error)

    # Fix line length issues
    if "E501" in error_types:
        new_lines = []
        for _i, line in enumerate(lines):
            if len(line.rstrip()) > 88:
                fixed_lines = fix_line_length(line.rstrip())
                new_lines.extend([l + "\n" for l in fixed_lines])
            else:
                new_lines.append(line)
        lines = new_lines

    # Write back the fixed content
    with open(file_path, "w") as f:
        f.writelines(lines)

    # Run ruff format
    subprocess.run(["ruff", "format", file_path], check=False, capture_output=True)


def main() -> None:
    """Main function to fix all Python files in the umap directory."""
    umap_dir = Path("/home/georgepearse/umap/umap")

    # Get all Python files
    python_files = list(umap_dir.glob("**/*.py"))

    for file_path in python_files:
        if "test" not in str(file_path):  # Skip test files for now
            fix_file(str(file_path))


if __name__ == "__main__":
    main()
