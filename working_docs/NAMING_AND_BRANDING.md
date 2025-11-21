# Squeeze/Reductio: Naming and Branding Guide

## Primary Names

### **Squeeze** (Preferred)
The primary short name for the research platform.

**Rationale**:
- Direct reference to "squeezing" high-dimensional data into lower dimensions
- Short and memorable
- Easy to pronounce and spell
- Modern and approachable
- Evokes the core concept intuitively

**Usage**: Marketing, documentation, general reference

**Import alias** (future):
```python
import squeeze as sq

# Then use:
pipeline = sq.DRPipeline([...])
embedding = sq.UMAP(...)
evaluator = sq.DREvaluator()
```

---

### **Reductio** (Alternative/Full)
The full formal name for academic or formal contexts.

**Rationale**:
- Reference to "reductio ad absurdum" (mathematical proof by reduction)
- Emphasizes the research platform aspect
- Suitable for academic papers and formal documentation
- Professional and scholarly tone

**Usage**: Academic papers, formal documentation, alternative branding

---

## Current State (Transition Period)

The package is currently named `umap` in the codebase but includes Squeeze/Reductio research platform capabilities.

### Current Imports:
```python
# Current (UMAP-based)
from umap import UMAP
from umap.composition import DRPipeline, EnsembleDR
from umap.metrics import DREvaluator
from umap.sparse_ops import SparseUMAP

# Will transition to in future:
import squeeze as sq
sq.DRPipeline([...])
sq.UMAP(...)
sq.metrics.DREvaluator()
sq.sparse.SparseUMAP()
```

---

## Future Transition Plan

### Phase 1 to Phase 2 Transition
When transitioning to the next major version:

1. **Package Rename**: `umap` → `squeeze`
2. **Module Restructure** (optional):
   - `squeeze.core` - Core UMAP algorithm
   - `squeeze.composition` - DRPipeline, EnsembleDR, etc.
   - `squeeze.metrics` - Evaluation framework
   - `squeeze.sparse` - Sparse operations
   - `squeeze.benchmark` - Benchmarking system

3. **Backward Compatibility**: Provide compatibility shim:
   ```python
   # umap package for backward compatibility
   from squeeze import UMAP, DREvaluator, DRPipeline
   ```

4. **Documentation**: Update all examples to use `squeeze`

---

## Rationale Summary

| Aspect | Squeeze | Reductio |
|--------|---------|----------|
| Length | Short (1 syllable) | Longer (4 syllables) |
| Memorability | Very high | High |
| Intuitiveness | Very intuitive | Scholarly |
| Use Case | General users, code | Academia, papers |
| Pronunciation | Intuitive | May need help |
| Modern feel | Yes | No |
| Professional feel | Good | Excellent |

---

## Brand Promise

Squeeze enables researchers to:
1. **Compose** multiple dimension reduction techniques
2. **Evaluate** embedding quality systematically
3. **Benchmark** algorithms and parameter configurations
4. **Compare** approaches on the same metrics
5. **Research** new dimensionality reduction methods

All with the power and reliability of UMAP at its core.

---

## Co-Branding Strategy

The project can be referred to as:
- **"Squeeze"** - For public/user-facing materials
- **"Squeeze (Reductio)"** - When introducing the project
- **"Reductio"** - In academic papers or formal contexts
- **"UMAP Research Platform"** - When emphasizing UMAP foundation

Example taglines:
- "Squeeze: Dimension Reduction Research Platform"
- "Reductio: Composing and Evaluating Dimensionality Reduction Techniques"
- "Squeeze - Built on UMAP, Extended for Research"

---

## Logo and Visual Identity (Future)

Suggested visual elements:
- **Icon**: Concentric circles or funnel shape (representing dimension reduction)
- **Color Palette**: Professional blue-green (science/research oriented)
- **Typography**: Modern, clean sans-serif font
- **Tagline**: "Reduce. Compose. Evaluate. Benchmark."

---

## Decision Timeline

- **Now**: Use Squeeze as primary name in documentation
- **Phase 2**: Plan package rename transition
- **v2.0**: Execute full package rename to `squeeze`
- **v2.1+**: Establish Squeeze as primary brand, Reductio as formal alternative

---

## Additional Notes

The name "Squeeze" works well because:
1. It's a verb (action-oriented)
2. It's a common word (low barrier to entry)
3. It directly describes what dimensionality reduction does
4. It's flexible for compound names:
   - "Squeeze Pipeline" (DRPipeline)
   - "Squeeze Metrics" (evaluation framework)
   - "Squeeze Benchmark" (benchmarking system)
   - "Squeeze Sparse" (sparse data support)

This makes it work as both a single-word brand and as a compound noun prefix.
