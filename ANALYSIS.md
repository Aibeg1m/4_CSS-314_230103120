# Cache Line & False Sharing — Analysis (ILLUSTRATIVE EXAMPLE)

**Student GitHub handle:** Aibeg1m  
**Status:** EXAMPLE ONLY — the following example numbers are **not measured on the student's CPU**. To create the submission-ready version, run `python false_sharing_lab.py` locally. The program will overwrite this file and generate `cache_lab_manifest.json` from actual telemetry.

## Q1 — MESI State Transitions

With stride = 1, thread 0 writes `buffer[0]` and thread 1 writes `buffer[1]`. These two 8-byte counters are in the same 64-byte cache line. If Core 0 holds the line in Exclusive (E), its store changes the line to Modified (M). Core 1 must obtain ownership through a Read-For-Ownership request. Core 0 relinquishes the line (M to I), and Core 1 receives it in M. Another store by Core 0 reverses ownership (Core 1 M to I and Core 0 I to M). Shared (S) copies, if present, must also be invalidated. Ownership transfer can delay the write and put pressure on the store buffer.

## Q2 — Timing and Coherency Overhead (illustrative values, NOT measured)

| Layout | Example elapsed time (ns) | Example ns/update |
|---|---:|---:|
| stride = 1 | 4,800,000,000 | 20.000 |
| stride = 4 | 3,000,000,000 | 12.500 |
| stride = 8 | 1,200,000,000 | 5.000 |
| stride = 16 | 1,100,000,000 | 4.583 |
| private baseline | 200,000,000 | 0.833 |

Example slowdown = 4,800,000,000 / 1,200,000,000 = **4.0×**. Total operations = four threads × 60,000,000 iterations = 240,000,000. These numbers are hypothetical and must not be represented as collected data.

## Q3 — 64-byte vs 128-byte spacing

In the **illustrative example**, stride 8 takes 1.2 s and stride 16 takes 1.1 s. Both place counters in separate cache lines and avoid direct false sharing. Some CPUs prefetch adjacent 64-byte lines, so 128-byte spacing can sometimes reduce secondary cache traffic. Adjacent-line prefetching alone does not imply an invalidation, and the difference depends on the processor and scheduling. The actual effect must be evaluated using the real timing results.

## Q4 — Compiler register promotion and LLVM

Without compiler constraints, a compiler may load a value once, accumulate in a register and store it only at the end, making a false-sharing benchmark misleading. The supplied `worker_strided_accumulate` uses Numba's `nogil=True` JIT with a side-effecting compiler memory barrier before and after an explicit array read-modify-write. Its optimized LLVM is emitted as `worker_strided_llvm.ll`. Inspect the generated loop to confirm `load i64` and `store i64` remain inside the loop. Exact LLVM instruction counts must come from the execution-generated manifest.

## Q5 — CPython GIL versus Numba nogil

In conventional GIL-enabled CPython, Python-bytecode counter loops do not execute simultaneously on separate CPU cores. Adding Python threads increases locking and scheduling overhead instead of exposing multi-core cache-coherence costs. Numba `@njit(nogil=True)` compiles native code and releases the GIL, allowing concurrent writes from separate threads and making false sharing observable. Free-threaded CPython configurations are an exception.

---

**Before submission:** run `python false_sharing_lab.py` on the intended host and upload its generated `ANALYSIS.md` and `cache_lab_manifest.json`, not these illustrative placeholders.
