# Results

Status of the pipeline as of **2026-09-17**. Every figure below is read from a recorded
artifact. Items still in progress are labelled as such rather than estimated.

---

## Hardware

Training and deployment run on deliberately different hardware. The split is the point:
train where the memory bandwidth is, deploy where the power budget is.

| Role | Platform | Specification |
| --- | --- | --- |
| **Training** | NVIDIA RTX PRO 6000 Blackwell (Molab) | 96 GB GDDR7, CUDA, BF16 |
| **Deployment target** | Intel Core Ultra 7 265T (Arrow Lake) | 20 cores, 32 GiB DDR5-5600, 512 GiB, Windows 11, bare metal |
| **Execution units** | Intel Core Ultra | CPU · integrated GPU · NPU |

All three Intel execution units are exposed to the OpenVINO runtime:

```json
{"available_devices": ["CPU", "GPU", "NPU"]}
```

---

## Validated

### 1. Trained policy checkpoint

Fine-tuning produced a complete, loadable checkpoint at step 4,000:

```json
{
  "complete": true,
  "missing": [],
  "weights_bytes": 906712552,
  "tokenizer_present": true
}
```

906.7 MB across eight files — weights, tokenizer, preprocessor and postprocessor.
**Progress: 4,000 / 6,826 configured steps (58.6%).**

### 2. Inference on Intel hardware

The policy loads and emits a correctly shaped action on the deployment target:

```
Loading HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights ...
Reducing the number of VLM layers to 16 ...
{'action_shape': [1, 12], 'action': [[0.1317, 0.0736, -0.0793, -0.2389, 0.0566, 0.2285,
  -0.1778, 0.2175, -0.4578, -0.5175, 0.1445, -0.2770]]}
```

`action_shape: [1, 12]` — the correct width for the twelve-joint bimanual interface.

### 3. Demonstration dataset

| Metric | Value |
| --- | --- |
| Successful episodes | 71 |
| Frames | 218,384 |
| Frame rate | 25 Hz |
| Cameras | 4 × 320×320×3 |
| State / action | 12 / 12 |

### 4. Contact-only grasping

Objects are held by friction alone. The jaws are position servos that drive into an
object and stall against it; the resulting contact force is the grip. There is no weld
constraint, no equality attachment, no object-follow and no pose teleport anywhere in
the transport path. Objects carry real mass, inertia and gravity, and a grasp is accepted
only when the object's measured centre rises by 20–25 mm or more.

---

## Evaluation batch — 10 seeds

A ten-seed sweep was run with appearance randomization active and a distinct language
paraphrase per seed. This measures the **demonstration generator's** robustness across
initial conditions — an internal data-quality metric, not a policy result.

| Outcome | Count |
| --- | --- |
| Completed successfully | 1 |
| Physics instability | 7 |
| Task check failed, physics stable | 2 |

The drawer pull succeeded in **all ten** runs (`drawer_qpos` 0.1252–0.1300, above the
80%-of-travel threshold). Failures occurred downstream of the drawer.

Failures separate into two mechanisms:

- **Early-onset joint divergence (3 runs).** A robot joint diverges during the drawer
  pull or first grasp. Reproducible per seed.
- **Late-stage object ejection (4 runs).** A single object diverges during a late
  placement stage. Arm joints remain within 1.0 rad throughout.

Full diagnostics, including per-seed onset steps and divergence sites, are in the
companion analysis. The generator is a data-production tool: its robustness determines
dataset yield, and it is being improved before further collection.

> **Policy evaluation has not been run.** No success rate, return, or task-completion
> figure for the trained SmolVLA policy exists yet. Nothing in this document should be
> read as a policy result.

---

## Benchmarking — CPU / GPU / NPU

**Status: measurement in progress.**

The target is to characterise the policy across all three Intel execution units, and to
report energy as well as latency. Throughput alone is misleading on a power-constrained
edge part: the fastest device is not always the one that can be sustained.

### Metrics

| Metric | Source | Why it is reported |
| --- | --- | --- |
| Throughput (inferences/s) | Timed loop, explicit iteration count | Raw wall-clock capability |
| Mean latency (ms/action) | Same loop, chunk ÷ 50 | Compared against the **40 ms** real-time control budget |
| Package power (W) | Package energy counter, sampled around the same loop | Sustained draw — the number that decides an edge deployment |
| Energy per inference (mJ) | Counter delta ÷ iterations | The cost of one decision |
| Efficiency (inferences/s/W) | Derived from the two above | Throughput per watt; decides the deployment target |

**Power has to come from a counter, not a datasheet.** The harness reads an Intel
package energy counter through the RAPL powercap interface where the host exposes one,
and samples it around the same timed loop as the latency — so power and latency describe
the same calls rather than two different runs. The Windows deployment target exposes no
unprivileged equivalent. There the harness records `"package_power_w": null`, prints
`power: not measured`, and leaves the energy columns empty. Filling them needs an
external meter or a Linux run of the same harness; a TDP figure is not a substitute.

### Real-time budget

The environment runs at a 25 Hz control rate. Each episode's `meta` array records
`[0.004, 0.04]` — a 0.004 s physics timestep and a **0.04 s control period**. A policy
inference must therefore complete inside **40 ms** to drive the robot in real time. This
is the threshold every device is measured against.

### Results

Measurements will be added here once the sweep completes. The harness reports an
explicit iteration count with every figure, so each number carries its own denominator,
and it names its power source so an empty energy column is legible as *not measured*
rather than as zero.

| Device | Throughput (inf/s) | Latency (ms/action) | Package power (W) | Energy (mJ/inf) | Efficiency (inf/s/W) | Meets 40 ms |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| GPU | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| NPU | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

Two outcomes are possible for the NPU and both are worth reporting: it may deliver the
best performance-per-watt of the three, or it may be unable to host a 500M-parameter
vision-language model with an iterative action expert. A negative result is still a
result, and would itself determine the deployment target.

> An earlier timing file recorded a single elapsed value with no iteration count. It is
> deliberately **not** quoted here, because a latency without a denominator is not a
> measurement.

---

## In progress

| Item | State |
| --- | --- |
| SmolVLA fine-tune | 4,000 / 6,826 steps |
| OpenVINO IR conversion | Not started — device detection only |
| CPU / GPU / NPU benchmark sweep | Harness ready, measurements pending |
| Generator stability | 7 of 10 seeds diverge; under investigation |
| Rendered video capture | Not started |
| Policy evaluation | Not started |

---

## Reproducing

```bash
python tests/test_repo_structure.py     # structure; runs anywhere
python scripts/task_demo.py             # physics; needs MuJoCo, no GPU
python benchmark_intel_devices.py --help # device sweep; run on the Intel target
```

See [`REPRODUCE.md`](REPRODUCE.md) for the full pipeline.
