# CuQSim — CUDA Quantum State-Vector Simulator

[[https://colab.research.google.com/assets/colab-badge.svg]](https://colab.research.google.com/drive/1Yc6quHutP5Lqwdly_1JjSjCD8-IfrZF-?usp=sharing)

**CuQSim** is a lightweight, zero-dependency GPU quantum circuit simulator written in CUDA C++ from scratch. It simulates $n$-qubit quantum circuits using $O(2^n)$ device memory, verifies correctness on entangled GHZ states, and benchmarks single-qubit gate throughput against both a CPU baseline and [NVIDIA cuQuantum (`cuStateVec`)](https://developer.nvidia.com/cuquantum-sdk).

---

## Key Features

- **Custom CUDA Gate Kernels (`kernels.cu`):**
  - `apply_gate_1q`: Applies arbitrary $2 \times 2$ single-qubit unitaries (`H`, `X`, `Z`, `Rx`, etc.) in `complex64` without allocating $2^n \times 2^n$ matrices.
  - `apply_cnot`: Applies controlled-NOT gates using bitwise control-qubit masking.
- **GHZ State Verifier (`main.cu`):** Constructs an $n$-qubit GHZ state $\frac{1}{\sqrt{2}}(\vert{}0\dots0\rangle + \vert{}1\dots1\rangle)$ and verifies exact state probabilities ($P(\vert{}0\dots0\rangle) = P(\vert{}1\dots1\rangle) = 0.5$).
- **Hardware-Aware Benchmarking:** Measures kernel execution time with `cudaEvent_t`, effective DRAM bandwidth (GB/s), speedup vs. single-threaded CPU, and runtime parity against NVIDIA `cuStateVec`.

---

## Architecture

For an $n$-qubit state vector of $2^n$ complex amplitudes, each gate kernel launches $2^{n-1}$ GPU threads. Thread `tid` maps directly to a pair of basis states `(idx0, idx1)` that differ only at bit `target`:

```cpp
unsigned long long low_mask = (1ULL << target) - 1;
unsigned long long idx0 = ((tid & ~low_mask) << 1) | (tid & low_mask);
unsigned long long idx1 = idx0 | (1ULL << target);
