# Multi-category dataset audit - MVTec AD

- Generated: `2026-11-18T20:09:14+00:00`
- Dataset root: `transistor`
- Categories requested: 6
- Ready: transistor, cable, leather, bottle, capsule, grid
- Missing / failed: none

| Category | Data present | Audit ready | Train good | Test good | Test anomalous | Masks | Defects |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| transistor | yes | PASS | 213 | 60 | 40 | 40 | bent_lead, cut_lead, damaged_case, misplaced |
| cable | yes | PASS | 224 | 58 | 92 | 92 | bent_wire, cable_swap, combined, cut_inner_insulation, cut_outer_insulation, missing_cable, missing_wire, poke_insulation |
| leather | yes | PASS | 245 | 32 | 92 | 92 | color, cut, fold, glue, poke |
| bottle | yes | PASS | 209 | 20 | 63 | 63 | broken_large, broken_small, contamination |
| capsule | yes | PASS | 219 | 23 | 109 | 109 | crack, faulty_imprint, poke, scratch, squeeze |
| grid | yes | PASS | 264 | 21 | 57 | 57 | bent, broken, glue, metal_contamination, thread |

## Per-category findings

- **transistor**: no critical issues.
- **cable**: no critical issues.
- **leather**: no critical issues.
- **bottle**: no critical issues.
- **capsule**: no critical issues.
- **grid**: no critical issues.

Full per-category audit details are stored under `reports/metrics/<category>/dataset_audit.md`.
Missing categories are reported, never simulated: no synthetic dataset samples are generated.
