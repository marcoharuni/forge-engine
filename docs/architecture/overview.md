# Overview

ForgeEngine separates its future Rust gateway and control-plane processes from Python engine
workers. They communicate through versioned contracts under `proto/forge/v1/`; Python is not
embedded in the gateway. Optional Triton, CUDA, CUTLASS, CuTe C++, and CuTe DSL paths remain
behind explicit capability and build boundaries.

TODO: Expand component ownership and lifecycle diagrams as interfaces stabilize.
