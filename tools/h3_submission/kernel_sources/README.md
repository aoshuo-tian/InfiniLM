# Historical QK NineToothed graph sources

`qknorm/` and `qkrope/` contain the final generator and graph-expression files from SHA-verified 2026-10-22 validation archives. These are not instructions to resume the historical experiments. QK+RoPE remains disabled in the selected build.

The source-build path uses the already generated gfx936 C/C++ inputs in `support/dependencies/`. It does not execute these generators. Regenerating them requires the matching NineToothed compiler/DTK/Triton and the historical compiler adaptation; unmodified PyPI packages are not asserted to produce byte-identical results. No regeneration or new GPU validation was performed during this GitHub packaging.

See `h3_submission/kernel-source-provenance.json` for original evidence-archive and per-file SHA256 identities.
