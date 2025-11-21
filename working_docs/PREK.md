# prek Configuration

This project uses **prek**, a Rust-based pre-commit framework that's significantly faster than the original Python-based pre-commit.

## Installation

### One-time Setup

Install prek globally using uv:

```bash
uv tool install prek
```

Alternatively, you can install via:
- `pip install prek` or `pipx install prek`
- `brew install prek` (macOS/Linux)
- `npm install -D @j178/prek` (npm)

### Initialize Hooks

After cloning the repo, initialize prek hooks:

```bash
prek install
prek install-hooks
```

This sets up the pre-commit hook that automatically runs before each commit.

## Running Hooks

### Automatically (on commit)
Hooks run automatically when you commit. To skip (not recommended):
```bash
git commit --no-verify
```

### Manually

Run all hooks on changed files:
```bash
prek run
```

Run all hooks on all files:
```bash
prek run --all-files
```

Run specific hooks:
```bash
prek run ruff-check
prek run ruff-format
prek run mypy
```

## Tools Configured

### 1. Ruff (Linter & Formatter)
- **All rules enabled** (`select = ["ALL"]`)
- **Auto-fixes**: Ruff will automatically fix many issues
- **Configuration**: See `pyproject.toml` under `[tool.ruff]`

**Key ignored rules:**
- `ANN`: Type annotations (handled by mypy type checker)
- `D100, D104, D105`: Module/package docstrings (too strict for scientific code)
- `E501`: Line length (handled by formatter)
- `BLE001`: Blind except (sometimes necessary)

**Per-file ignores:**
- Test files: Additional docstring rules + `S101` (assert usage)
- setup.py: Missing module docstring

### 2. ty (Type Checker) - Via Pre-commit Local Hook
- **Rust-based type checker** by Astral - significantly faster than Python-based alternatives (0.5s vs mypy's 18s+)
- **Strict mode enabled** - enforces type annotations
- **Configuration**: See `pyproject.toml` under `[tool.ty]`
- **Automatic detection** - Automatically detects virtual environments and reads config
- **Integrated via local hook** - Runs automatically on commit as part of prek

**Running ty:**
```bash
# Automatically (on commit or via prek)
prek run ty

# Manually
ty check
ty check path/to/file.py
```

**Configuration (pyproject.toml):**
```toml
[tool.ty]
python_version = "3.9"
strict = true
```

**Note**: ty is pre-alpha but actively developed by Astral. While there's no official pre-commit hook repository yet, we use a local hook definition in `.pre-commit-config.yaml` for seamless integration.

### 3. Pre-commit Hooks
Standard hooks for code hygiene:
- Trailing whitespace removal
- EOF fixer
- YAML validation
- Large file checks (>5MB)
- AST syntax validation
- Merge conflict detection

## Ruff Rules Reference

Ruff has many rule categories. With `select = ["ALL"]` enabled, we use all available rules except those in the `ignore` list.

**Common rule prefixes:**
- `E`: pycodestyle (PEP 8 style violations)
- `F`: Pyflakes (logical errors)
- `W`: pycodestyle (warnings)
- `C`: mccabe (complexity)
- `I`: isort (import ordering)
- `D`: pydocstyle (docstrings)
- `S`: bandit (security)
- `ANN`: Annotations (type hints)
- `RUF`: Ruff-specific rules
- `UP`: pyupgrade (modernizing syntax)
- `B`: flake8-bugbear (bug detection)

See [Ruff Rules](https://docs.astral.sh/ruff/rules/) for the complete list.

## Configuration Files

### `.pre-commit-config.yaml`
Defines which tools run and their versions. This is the standard pre-commit format that prek uses directly (no conversion needed).

### `pyproject.toml`
Contains configuration for:
- **ruff**: Linter/formatter settings
- **mypy**: Type checker settings

## Troubleshooting

### Hooks fail with dependency errors
prek caches hook environments in `~/.cache/prek`. If you encounter issues:
```bash
prek cache clean
prek install-hooks
```

### ty reports type errors
These are genuine type errors that need to be fixed:
```bash
# View errors from local hook run
prek run ty

# Or run directly
ty check

# Fix by adding type annotations
# See ty output for specific locations
```

### Ruff reports too many issues
If you want to gradually adopt ruff, you can:
1. Fix issues incrementally
2. Temporarily add rules to ignore list
3. Run `prek run ruff-check -- --fix` to auto-fix many issues

### Can't commit due to large hooks
First time hook installation downloads dependencies. This may take a minute. Subsequent runs are fast.

## Advantages Over pre-commit

- **2-3x faster** execution time
- **~50% less disk space** usage
- **Centralized caching** (`~/.cache/prek` vs per-repo)
- **Automatic Python version management** via uv
- **Same config format** (`.pre-commit-config.yaml`)
- **Drop-in replacement** for pre-commit

## Documentation

- **prek**: https://prek.j178.dev/
- **ruff**: https://docs.astral.sh/ruff/
- **ty**: https://docs.astral.sh/ty/
- **pre-commit hooks**: https://pre-commit.com/

## GitHub Actions Integration

All tools are integrated into GitHub workflows:

### Workflow Steps
1. **Setup**: Install uv, Python, and dependencies
2. **Code Quality**: Run prek hooks (ruff + standard checks)
3. **Type Checking**: Run ty type checker
4. **Testing**: Run pytest with coverage
5. **Reporting**: Upload coverage to Codecov

The workflow runs on:
- **Triggers**: Push to master/develop and pull requests
- **Python versions**: 3.9, 3.10, 3.11, 3.12
- **Platform**: Ubuntu latest

### Workflow Failure Handling
- If prek hooks fail (ruff, code quality), the workflow fails
- If ty fails (type errors), the workflow fails
- Tests use fallback to full suite if testmon optimization fails
- Coverage upload is non-blocking

## Using ty (Pre-alpha)

We use `ty` (Astral's Rust-based type checker) from the start, even in its pre-alpha stage (v0.0.1-alpha.25):

**Benefits:**
- Significantly faster than mypy (0.5s vs 18s+)
- Consistent with the Astral ecosystem (ruff, uv, etc.)
- Automatically detects and uses virtual environments
- Simple, focused type checking
- Fully integrated with prek for automatic checking on commits

**Local Integration:**
```bash
# Automatically on commit (via prek)
git commit

# Or manually run
prek run ty
ty check
```

**Installation:**
```bash
uv tool install ty
```

As ty matures toward feature-completeness and stable release (planned late 2025), it will become the standard type checker across the Python ecosystem.
