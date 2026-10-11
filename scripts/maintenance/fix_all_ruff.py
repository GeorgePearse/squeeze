#!/usr/bin/env python3
"""Comprehensive script to fix all ruff errors in UMAP codebase."""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def get_ruff_errors(file_path: str) -> list[dict[str, Any]]:
    """Get all ruff errors for a file in JSON format."""
    result = subprocess.run(
        ["ruff", "check", file_path, "--output-format", "json"],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return []
    return []


def auto_fix_with_ruff(file_path: str) -> None:
    """Run ruff auto-fix with unsafe fixes."""
    subprocess.run(
        ["ruff", "check", file_path, "--fix", "--unsafe-fixes"],
        check=False,
        capture_output=True,
    )
    subprocess.run(
        ["ruff", "format", file_path],
        check=False,
        capture_output=True,
    )


def fix_imports(content: str) -> str:
    """Add necessary imports for type hints."""
    lines = content.splitlines()

    # Check if we need typing imports
    needs_typing = set()

    for line in lines:
        if "-> None:" in line or "-> None" in line:
            needs_typing.add("None")
        if "-> bool:" in line or "-> bool" in line:
            needs_typing.add("bool")
        if "-> int:" in line or "-> int" in line:
            needs_typing.add("int")
        if "-> float:" in line or "-> float" in line:
            needs_typing.add("float")
        if "-> str:" in line or "-> str" in line:
            needs_typing.add("str")
        if "Optional[" in line:
            needs_typing.add("Optional")
        if "List[" in line:
            needs_typing.add("List")
        if "Dict[" in line:
            needs_typing.add("Dict")
        if "Tuple[" in line:
            needs_typing.add("Tuple")
        if "Any" in line and ": Any" in line:
            needs_typing.add("Any")
        if "Union[" in line:
            needs_typing.add("Union")

    # Find where to insert imports
    import_line = -1
    last_import = -1
    for i, line in enumerate(lines):
        if line.startswith(("from typing import", "import typing")):
            import_line = i
            break
        if line.startswith(("import ", "from ")):
            last_import = i

    # Add typing imports if needed
    if needs_typing:
        typing_imports = ["Any", "Dict", "List", "Optional", "Tuple", "Union"]
        actual_imports = [imp for imp in typing_imports if imp in needs_typing]

        if import_line >= 0:
            # Update existing typing import
            lines[import_line] = (
                f"from typing import {', '.join(sorted(actual_imports))}"
            )
        elif last_import >= 0:
            # Add new typing import after other imports
            lines.insert(
                last_import + 1,
                f"from typing import {', '.join(sorted(actual_imports))}",
            )
        else:
            # Add at the beginning after docstring
            insert_pos = 0
            if lines and lines[0].startswith('"""'):
                for i, line in enumerate(lines[1:], 1):
                    if '"""' in line:
                        insert_pos = i + 1
                        break
            lines.insert(
                insert_pos,
                f"from typing import {', '.join(sorted(actual_imports))}",
            )

    return "\n".join(lines)


def add_function_annotations(content: str) -> str:
    """Add basic type annotations to functions."""
    lines = content.splitlines()

    for i, line in enumerate(lines):
        # Skip if already has annotation
        if "->" in line:
            continue

        # Check for function definitions
        if line.strip().startswith("def "):
            # Special cases for common patterns
            if "def __init__(self" in line and "):" in line and "->" not in line:
                lines[i] = line.replace("):", ") -> None:")
            elif ("def __str__(self" in line and "):" in line and "->" not in line) or (
                "def __repr__(self" in line and "):" in line and "->" not in line
            ):
                lines[i] = line.replace("):", ") -> str:")
            elif "def __len__(self" in line and "):" in line and "->" not in line:
                lines[i] = line.replace("):", ") -> int:")
            elif (
                ("def __bool__(self" in line and "):" in line and "->" not in line)
                or (
                    re.match(r"^\s*def is_\w+\(", line)
                    and "):" in line
                    and "->" not in line
                )
                or (
                    re.match(r"^\s*def has_\w+\(", line)
                    and "):" in line
                    and "->" not in line
                )
            ):
                lines[i] = line.replace("):", ") -> bool:")

    return "\n".join(lines)


def add_module_docstring(content: str) -> str:
    """Add module docstring if missing."""
    lines = content.splitlines()

    if not lines:
        return '"""Module docstring."""\n'

    # Check if first non-empty, non-comment line is a docstring
    first_code_line = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith("#"):
            first_code_line = i
            break

    if first_code_line < len(lines) and not lines[first_code_line].strip().startswith(
        '"""',
    ):
        # Get module name from file
        module_name = Path(sys.argv[1]).stem if len(sys.argv) > 1 else "module"
        docstring = f'"""Module for {module_name} functionality."""\n\n'
        lines.insert(first_code_line, docstring)

    return "\n".join(lines)


def fix_line_length(content: str) -> str:
    """Fix lines that are too long."""
    lines = content.splitlines()
    fixed_lines = []

    for line in lines:
        if len(line) <= 88:
            fixed_lines.append(line)
            continue

        # Handle docstrings
        if '"""' in line or "'''" in line:
            indent = len(line) - len(line.lstrip())
            words = line.strip().split()
            current = " " * indent
            result = []

            for word in words:
                if len(current) + len(word) + 1 <= 88:
                    if current.strip():
                        current += " " + word
                    else:
                        current += word
                else:
                    if current.strip():
                        result.append(current)
                    current = " " * indent + word
            if current.strip():
                result.append(current)
            fixed_lines.extend(result)

        # Handle import statements
        elif line.strip().startswith("from ") and "import" in line:
            if "," in line:
                parts = line.split("import")
                if len(parts) == 2:
                    base = parts[0] + "import ("
                    imports = parts[1].strip().split(",")
                    indent = len(line) - len(line.lstrip()) + 4
                    fixed_lines.append(base)
                    for imp in imports[:-1]:
                        fixed_lines.append(" " * indent + imp.strip() + ",")
                    fixed_lines.append(" " * indent + imports[-1].strip())
                    fixed_lines.append(")")
            else:
                fixed_lines.append(line)

        # Handle function calls and definitions
        elif "(" in line and ")" in line and "," in line:
            # Try to break at commas
            indent = len(line) - len(line.lstrip())
            # Find the opening parenthesis position
            paren_pos = line.index("(")
            base = line[: paren_pos + 1]
            rest = line[paren_pos + 1 :]

            if ")" in rest:
                close_pos = rest.rindex(")")
                args = rest[:close_pos]
                after = rest[close_pos:]

                if "," in args:
                    arg_list = args.split(",")
                    fixed_lines.append(base)
                    new_indent = " " * (indent + 4)
                    for arg in arg_list[:-1]:
                        fixed_lines.append(new_indent + arg.strip() + ",")
                    fixed_lines.append(new_indent + arg_list[-1].strip() + after)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        # For other long lines, try to break at operators
        elif " and " in line:
            parts = line.split(" and ")
            indent = len(line) - len(line.lstrip())
            fixed_lines.append(parts[0] + " and")
            for part in parts[1:]:
                fixed_lines.append(" " * (indent + 4) + part)
        elif " or " in line:
            parts = line.split(" or ")
            indent = len(line) - len(line.lstrip())
            fixed_lines.append(parts[0] + " or")
            for part in parts[1:]:
                fixed_lines.append(" " * (indent + 4) + part)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def remove_commented_code(content: str) -> str:
    """Remove commented-out code lines."""
    lines = content.splitlines()
    fixed_lines = []

    for line in lines:
        stripped = line.strip()
        # Keep legitimate comments (starting with # and a space or special markers)
        if stripped.startswith("#"):
            # Check if it looks like code
            if (
                stripped[1:]
                .strip()
                .startswith(
                    (
                        "import ",
                        "from ",
                        "def ",
                        "class ",
                        "if ",
                        "for ",
                        "while ",
                        "return ",
                        "print(",
                        "self.",
                        "super(",
                    ),
                )
            ):
                continue  # Skip commented-out code
            # Keep type: ignore and other special comments
            if (
                "type:" in stripped
                or "noqa" in stripped
                or "pylint" in stripped
                or stripped.startswith("# ")
                or stripped == "#"
            ):
                fixed_lines.append(line)
            else:
                # Likely commented code, skip it
                continue
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_variable_names(content: str) -> str:
    """Fix variable naming convention issues."""
    lines = content.splitlines()

    # Common replacements for variable names
    replacements = {
        r"\bN\b": "n",
        r"\bD\b": "d",
        r"\bX\b": "x",
        r"\bY\b": "y",
        r"\bZ\b": "z",
        r"\bA\b": "a",
        r"\bB\b": "b",
        r"\bC\b": "c",
        r"\bM\b": "m",
        r"\bP\b": "p",
        r"\bQ\b": "q",
        r"\bR\b": "r",
        r"\bS\b": "s",
        r"\bT\b": "t",
        r"\bU\b": "u",
        r"\bV\b": "v",
        r"\bW\b": "w",
    }

    for i, line in enumerate(lines):
        # Skip if it's in a comment or string
        if "#" in line:
            code_part = line[: line.index("#")]
            comment_part = line[line.index("#") :]
        else:
            code_part = line
            comment_part = ""

        # Apply replacements only in specific contexts
        if "def " in code_part or "for " in code_part or " = " in code_part:
            for old, new in replacements.items():
                # Only replace in variable contexts, not in strings or as part of larger words
                code_part = re.sub(old + r"(?=[\s,\)\]:])", new, code_part)

        lines[i] = code_part + comment_part

    return "\n".join(lines)


def process_file(file_path: str) -> None:
    """Process a single Python file to fix ruff errors."""
    # First, run auto-fix
    auto_fix_with_ruff(file_path)

    # Read the file
    with open(file_path) as f:
        content = f.read()

    # Apply custom fixes
    content = add_module_docstring(content)
    content = add_function_annotations(content)
    content = fix_imports(content)
    content = fix_line_length(content)
    content = remove_commented_code(content)
    # Skip variable name fixing for now as it might break functionality
    # content = fix_variable_names(content)

    # Write back
    with open(file_path, "w") as f:
        f.write(content)

    # Run auto-fix again
    auto_fix_with_ruff(file_path)

    # Check remaining errors
    errors = get_ruff_errors(file_path)
    if errors:
        for _error in errors[:5]:  # Show first 5 errors
            pass
    else:
        pass


def main() -> None:
    """Main function to process all Python files."""
    umap_dir = Path("/home/georgepearse/umap/umap")

    # Get all Python files (excluding tests initially)
    python_files = sorted(umap_dir.glob("*.py"))

    for file_path in python_files:
        if file_path.name != "__init__.py":  # Skip already fixed file
            process_file(str(file_path))

    # Now process test files with appropriate handling
    test_files = sorted(umap_dir.glob("tests/*.py"))

    for file_path in test_files:
        process_file(str(file_path))


if __name__ == "__main__":
    main()
