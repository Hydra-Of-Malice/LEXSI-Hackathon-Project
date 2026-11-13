## Summary

`circuit_info_from_node_scores` keeps every layer that holds any top-fraction node. Top nodes are spread through a model, so a top-20% node set can select nearly every layer (30 of 36 on Tiny Aya in our runs), and a circuit-restricted edit then restricts little. This adds an opt-in `layer_fraction=` that ranks layers by total absolute node score instead. Existing results don't change.

## Pillar / method affected

Interpret: `safetune.core.circuit_kit` (the CircuitKIT adapter). Its `layer_subset` is read by C-ΔΘ (`apply_ctheta`, `sweep_ctheta_strength`) and SafetyLoRA.

## Type of change

- [x] Bug fix (non-breaking)
- [ ] Faithfulness correction (method now matches its cited paper)
- [ ] New method / benchmark
- [ ] Breaking change
- [ ] Documentation
- [ ] Other

## What changes

- `layer_fraction=` on `circuit_info_from_node_scores`, `load_circuit_info_from_file` and `get_circuit_info`. It keeps the top fraction of layers by total absolute node score, and `top_fraction` then picks among the nodes in those layers.
- New `layer_mass(node_scores)` (total absolute score per layer, so opposite-signed nodes add up instead of cancelling) and `top_layers(node_scores, layer_fraction)`. Both are exported from `safetune.core.circuit_kit`.
- With `layer_fraction` set, a file that has `node_scores` gets its `safety_units` and `layer_suggestions` re-derived. CircuitKIT 0.2 writes those keys beside `node_scores` with the unranked selection, and they win when the file is read, so without this a real `*_scores.json` would keep the unranked layers. A file without `node_scores` keeps its keys and logs a warning.
- The file validation added in #50 still runs first. `get_circuit_info`'s docstring and the CircuitInfo page now say it reads `layer_fraction` as well as `top_fraction`.
- Docs: `circuit-info-object.md` (function table and a short section). CHANGELOG entry under `[Unreleased]` / Fixed.

Left for you to decide: whether `layer_fraction` should become the default, and whether CircuitKIT's `interop_fields` should rank layers too (it writes the same unranked `layer_subset`).

## Audit badge change (if applicable)

None.

## Checklist

- [x] Code follows the repo style (black, isort, line-length 100). The new code is black- and isort-clean; the touched files already differ from black on `main`, and those lines are left alone.
- [x] Tests added / updated for the change
- [ ] `pytest tests/` passes locally. Only the related files were run (below), not the full suite.
- [x] `docs/trust/feature-map.md` updated if a badge changed (no badge changed)
- [x] `CHANGELOG.md` updated under `[Unreleased]`
- [x] No secrets / credentials committed

## Tests

Windows 11, Python 3.12, `pip install -e ".[dev]"`, branch based on `main` at `263ea24`. Each command was run twice with the same result.

- `pytest tests/safety/test_circuit_kit.py`: 19 passed. 17 tests are new, and 16 of them fail against `main`'s source.
- `pytest tests/safety/test_circuit_kit.py tests/runtime/test_interop.py tests/diagnostic/test_recover_ctheta.py tests/safety/test_safety_lora.py`: 50 passed (33 on `main`).

The test scores are synthetic, shaped like the Tiny Aya case: 36 layers of 16 heads and an MLP, where 7 layers hold nearly all the score and 23 others hold one node just inside the top 20%.

Rebased on 13 November onto `main` at `cc3de5f`, with no conflicts. Re-run there on macOS, Python 3.12: `pytest tests/safety/test_circuit_kit.py tests/runtime/test_interop.py`: 43 passed.

## Licence

This contribution is made under SafeTune's LSAL v1.2. As `LICENSE.md` section 6 and `CONTRIBUTING.md` say, it grants Lexsi Labs a perpetual, worldwide, royalty-free right to use, modify, distribute and license it under any terms, including commercial ones, and I have the right to submit it.
The code was first written for our MIT-licensed project langsafe, and both of its copyright holders (Aditya Arnav and Pallab Das) agree to this grant.
