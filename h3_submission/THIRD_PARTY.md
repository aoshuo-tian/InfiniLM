# Third-party source attribution

Original InfiniTensor project license and pinned Git submodule references remain in the repository.
Generated NineToothed host launchers / ABI derive from InfiniTensor NineToothed 0.25.0; see the upstream Apache-2.0 license supplied with the ABI (and the Core submission directory). Device code embedded in generated sources is reused for gfx936, not a new handwritten CUDA/HIP kernel.

The matching InfiniLM repository supplies selected build headers:
- NVIDIA CCCL 2.4.0 Thrust headers: `support/tooling/cccl-2.4.0/LICENSE` (retained original exceptions).
- pybind11 3.1.0 headers: `support/tooling/pybind11/LICENSE`, obtained from the upstream v3.1.0 tag.
- GCC OpenMP header: `support/tooling/openmp/COPYRIGHT`, original notices in `include/omp.h` retained.

Official vendor PyTorch, DTK/RCCL, FlashAttention binary, model weights and Python dependency archives are external requirements and are not redistributed in this source commit. See [REVIEWER_COMMANDS_20261111.md](REVIEWER_COMMANDS_20261111.md) for versions and scope.
