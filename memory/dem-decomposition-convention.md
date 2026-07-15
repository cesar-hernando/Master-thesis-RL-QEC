---
name: dem-decomposition-convention
description: Always decompose DEMs with decompose_errors_for_stim_surface_code_coords, not stim's decompose_errors=True
metadata:
  type: feedback
---

Across the entire codebase, build decomposed DEMs with `decompose_errors_for_stim_surface_code_coords` (from `src/NeuralCM/decompose_errors.py`) applied to `circuit.detector_error_model()`. Do NOT use stim's built-in `circuit.detector_error_model(decompose_errors=True)`.

**Exception:** scripts/notebooks that deliberately compare the two decomposition methods may use both.

**Why:** the custom (tesseract-style) decomposition forces each decomposed component to trigger detectors of a single basis (X or Z), which is the decomposition the correlated matcher's correlation statistics (`corr_tracer`, line graph) assume. Stim's heuristic decomposition gives different, slightly worse correlation structure for CM. See [[decoder-naming-and-ncm-coefficients]].

**How to apply:** `dem = decompose_errors_for_stim_surface_code_coords(circuit.detector_error_model())`. When refactoring, replace any `decompose_errors=True` with this unless the file's purpose is a decomposition comparison. Import it under its full name — the user does not want it aliased (no `as dec_coords`).
