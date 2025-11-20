# Agent Development Workflow

This document specifies the workflow for implementing tasks and features in this repository.

## Core Principles

Every task should follow a structured workflow to ensure code quality, traceability, and easy review.

## Project Scope Constraints

**GPU Implementation Policy:** We are **NOT pursuing GPU implementations** (CUDA/Metal/OpenCL) for this project. The focus is on CPU-based optimizations including SIMD vectorization, improved algorithms (RobustPrune), and better caching strategies. Any roadmap items or documentation mentioning GPU acceleration should be considered **out of scope** and de-prioritized.

## Workflow

### 1. Create a Fresh Branch

Before starting any task, create a new branch from `master`:

```bash
git checkout master
git pull origin master
git checkout -b <branch-name>
```

**Branch naming conventions:**
- Feature: `feat/<description>` (e.g., `feat/add-simd-vectorization`)
- Bug fix: `fix/<description>` (e.g., `fix/memory-leak`)
- Documentation: `docs/<description>` (e.g., `docs/update-installation`)
- Refactoring: `refactor/<description>` (e.g., `refactor/optimize-metric-computation`)
- Tests: `test/<description>` (e.g., `test/add-parametric-umap-tests`)

### 2. Implement the Task

Work on the implementation in your branch:

- Write code following the project's conventions
- Add tests for new functionality
- Update documentation as needed
- Run pre-commit hooks to ensure code quality
- Commit changes with clear, descriptive messages

**Commit Message Guidelines:**
- Use imperative mood ("add" not "adds", "fix" not "fixes")
- First line should be concise (50 chars or less)
- Provide detailed explanation in the body if needed
- Reference related issues if applicable

Example:
```bash
git add .
git commit -m "Add SIMD vectorization for distance metrics

- Implement AVX2/NEON optimized Euclidean distance
- Add runtime CPU feature detection
- Update benchmarks to measure SIMD impact
"
```

### 3. Submit a Draft PR

After completing the task, push your branch and create a **draft pull request**:

```bash
git push -u origin <branch-name>
gh pr create --draft --title "<title>" --body "<description>"
```

**Draft PR Requirements:**
- Clear, descriptive title
- Summary of changes made
- List of key implementation details
- Note any testing performed
- Flag any known issues or TODOs

Example:
```markdown
## Summary
Implements SIMD vectorization for distance metrics on CPU.

## Changes
- Added AVX2-optimized Euclidean distance computation
- Implemented NEON support for ARM processors
- Added runtime CPU feature detection and fallback

## Testing
- Unit tests pass on all CPU architectures
- Benchmarks show 3.2x speedup on large vectors (AVX2)
- Validated on x86_64 and ARM64 platforms
```

### 4. Code Review

The draft PR allows for:
- Early feedback on approach and design
- Discussion of implementation details
- Identification of issues before final submission
- Iteration based on review comments

When ready for final review, convert the draft to a regular PR or request review.

## Pre-submission Checklist

Before creating a PR, ensure:

- [ ] All tests pass locally
- [ ] Pre-commit hooks pass
- [ ] Code follows project conventions
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with `master`
- [ ] No unrelated changes are included

## Important Notes

### No Direct Commits to Master
- **Never commit directly to `master`**
- All changes must go through the branch → draft PR → review → merge workflow

### Keep Branches Fresh
- Rebase on `master` if it diverges significantly
- Keep branch scope focused on a single task
- Delete branch after merge

### Tests Are Required
- All new code must have corresponding tests
- All tests must pass before submitting PR
- Include both unit and integration tests where appropriate

### Documentation
- Update README if adding user-facing features
- Update docstrings for code changes
- Add migration guides for breaking changes

## Example Workflow

```bash
# 1. Create a branch
git checkout -b feat/add-inverse-transform-optimization

# 2. Implement the feature
# ... write code, tests, docs ...

# 3. Test locally
pytest umap/tests/ -v

# 4. Commit changes
git add .
git commit -m "Optimize inverse transform computation

- Use vectorized operations for faster calculation
- Add caching for repeated transforms
- Improve memory efficiency
"

# 5. Push to remote
git push -u origin feat/add-inverse-transform-optimization

# 6. Create draft PR
gh pr create --draft \
  --title "feat: Optimize inverse transform computation" \
  --body "
## Summary
Implements optimization improvements for inverse_transform method.

## Changes
- Vectorized operations for 2x speedup
- Caching layer for repeated queries
- Reduced memory allocation

## Testing
- All tests pass
- Benchmarks show 50% improvement on large datasets
  "
```

## Questions?

If you're unsure about any aspect of the workflow:
1. Check existing PRs for examples
2. Refer to the project's CLAUDE.md for additional conventions
3. Open an issue with questions about the process
