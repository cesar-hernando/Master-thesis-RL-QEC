# Memory index

- [CM weight clipping](cm-weight-clipping.md) — the binding CM discount clip is the implied_p cap (0.499999), not min/max weight; native CM ignores all these bounds.
- [Decoder naming & NCM coefficients](decoder-naming-and-ncm-coefficients.md) — CM=native enable_correlations, NCM=linear PPO agent (damped coeffs 0.33/-0.39/-0.39/-0.60); 2-pass conditional CM == native CM exactly.
- [DEM decomposition convention](dem-decomposition-convention.md) — always decompose DEMs with decompose_errors_for_stim_surface_code_coords, never stim's decompose_errors=True (unless deliberately comparing both).
