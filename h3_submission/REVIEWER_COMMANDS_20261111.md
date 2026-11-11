# MiniMax H3 public source reproduction

The submitted repositories are InfiniCore `minimax-h3`, InfiniLM `minimax_h3`,
and NineToothed `minimax_h3`. The report must pin their full commit identities.
The final report is the submission; runtime archives and AFS paths are not
alternatives to these public source repositories.

## Environment and assets

Use the specified Linux x86_64 Hygon image with Python 3.11, DTK 26.04 and
the existing vendor Torch distribution `2.9.0+das.opt1.dtk2604` (the imported
Torch module may report `2.9.0`). Do not replace vendor Torch, NumPy or drivers.
Use persistent writable storage outside all three code checkouts. Shared CPU
sessions may need several leases; preserve the build directory between them.
The framework stage resumes verified objects and never marks an interrupted
build complete. Review available disk space before staging sources and objects.

Supply the shared MiniMax-H3 `diffusers` model directory and the official
vendor wheel:
`flash_attn-2.6.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl`.
Its size is 658898543 bytes and its SHA256 is
`8546e46bcbfcd346757c67a0ff180ace2a49716d9724d52ee870d04b4c3b41d1`.
The helper checks both and extracts only the shared library; it does not pip
install this wheel. Model weights and vendor binaries are not committed.

The private dependency installer accepts `H3_WHEELHOUSE` for a populated Linux
CPython 3.11 wheel directory. When unset it uses the documented package indexes.
Nodes without public network require this offline directory in advance.
Inference requirements include Janus and xxhash declared by InfiniLM, the pinned
Transformers/Diffusers/Accelerate packages, PyAV and the separately SHA-verified
torchvision CPU wheel. Comparison dependencies have their own explicit stage.

## Seven stages

1. Clone all three public repositories, check out the report's full commits,
   and run `git submodule update --init --recursive` in Core and LM.
2. Export `H3_REPRO_ROOT`, `H3_FLASH_WHEEL` and optionally `H3_WHEELHOUSE`.
   Run `reviewer_reproduce.sh prepare`, `dependencies`,
   `comparison-dependencies`, and `flash` on shared CPU. Check the needed
   Transformers/Diffusers interfaces and comparison imports before GPU use.
3. In the shared CPU image, run `framework`, then `ccl`. After a lease expires,
   use the same root and run `framework` again; do not rerun `prepare` over it.
4. In the allocated eight-DCU image, run `verify` and `communication`.
   These verify actual package/library locations and direct 2/4/8-device CCL.
   Use matching base Python/Torch/NumPy identities between the CPU build image
   and the DCU image; the base-identity guard intentionally rejects a mismatch.
5. The frozen input is the included clear puppy image with SHA256
   `8dacfbb2bf2ca5593c8b33222b74e7c265380c8441a41bc09718083fe1ca3fd8`.
   The prompt is exactly `A dog falls asleep`; width 768, height 928, 124 frames,
   24 FPS, 50 requested steps, seed 42. There are 49 denoiser calls in the
   preserved sample. Both native engines use the same logical devices 0–7, TP8.
6. Run `native` in the eight-DCU allocation. The worker uses newly built native
   libraries; the old platform restorer is not an execution route. VAE, cast,
   mixed precision, vendor Flash and cache computation are unchanged.
7. Run `reference` in an eight-DCU allocation using the official PyTorch legacy
   path: text modules on 0–3, Transformer modules on 4–7 with `device_map`, not
   TP8. The original generator imports native modules even with `--legacy`;
   both routes verify the new native library identities. This import does not
   select native model computation for the legacy reference.
   In a CPU session run `compare` using the dependencies prepared in step 2; the
   unmodified original compare script receives both existing videos and skips
   generation. `H3_REFERENCE_VIDEO` can instead point to an existing matching
   reference. It must match the recorded input, model and parameters.

All commands use `bash InfiniLM/tools/h3_submission/reviewer_reproduce.sh STAGE`
from any directory, with an absolute path when outside the source parent.
The script never allocates resources and never automatically retries jobs.
Each generation process has a 1740-second limit, suitable for a 30-minute lease.
Use a new `H3_OUTPUT_ROOT` for a separately authorized run; existing worker
output directories are never overwritten.
Comparison also refuses an existing comparison directory or stdout log. For a
new comparison output root, explicitly supply `H3_REFERENCE_VIDEO` and
`H3_NATIVE_VIDEO` when the existing videos reside under the previous root.

CPU compilation, DCU generation and CPU comparison require the indicated
environments. Do not concatenate all stages into a shared CPU terminal and
expect it to switch to a DCU allocation. Re-export the same persistent root in
each new terminal. A lease may end between stages; one process's 1740-second
limit does not imply that several processes fit into the same 30-minute lease.

## What is compiled and what was validated

The build recompiles Core, Python extensions, CCL and InfiniLM host code and
links repository-supplied gfx936 device code. It does not claim regeneration of
every NineToothed kernel. The modified NineToothed compiler sources are public
with provenance in that repository. Installing a new Triton is not required
merely to link these preserved device objects.

The preserved round6 build plus package-link repair produced new native
libraries. Those libraries were actually imported, used for communication,
and used to generate the fixed full video in job
`ab614174ec4d4a2e8031b4541b4574bf`. This establishes that runtime, not an
untested one-shot run of every newly consolidated delivery command.

This consolidation has host configuration, source-identity and unchanged-worker
checks. The Linux relative-link test is skipped on Windows, where this account
lacks symlink privileges. The final public command sequence has not been rerun
from a fresh target build in this update. Do not state otherwise.

The report retains only the historical shared-TP8 cache result 82.468%, from
the original recorded runtime/video pair in `tested-result.json`; it is not
the score of the reconstructed build or a predicted outcome of these commands.
No latest local score is adopted in the report. Historical evidence is retained.
