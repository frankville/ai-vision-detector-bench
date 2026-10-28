# ai-vision-detector-bench

Benchmark and compare real-time object detection models on live video streams.

Choosing a detector for a camera deployment is three decisions at once, and most
comparisons only help with one of them. Published benchmarks report accuracy on
COCO. They do not tell you whether the model keeps up with a 12 fps RTSP stream
on the mini PC already installed at the site, how many cameras one box can serve
before it starts dropping frames, or whether the licence lets you ship it at all.

This tool answers the second and third questions on hardware you actually have,
using streams you actually run.

```
visionbench sweep -s rtsp://camera/stream --all-permissive -d 60
visionbench report --markdown
```

![Dashboard showing a live detection view with per-stage latency and resource metrics](docs/dashboard.png)

## What it measures

| Group | Metrics |
| --- | --- |
| Throughput | decoded fps, processed fps, dropped frames, drop ratio, real-time factor, estimated streams per host |
| Latency | preprocess, inference, postprocess and end-to-end, each with mean, p50, p95, p99 and max |
| Resources | process and system CPU, resident memory, GPU utilisation, GPU memory, power draw, temperature |
| Cost | GPU milliseconds and joules per processed frame, where the platform reports power |
| Output | detections per frame, per-class totals, mean confidence |

End-to-end latency is measured from the moment the decoder produced the frame to
the moment detections are ready, so it includes the time a frame spent waiting
for a busy consumer. That waiting time is invisible to a benchmark that times
only the forward pass, and it is most of what an alerting pipeline actually
feels.

## The two source modes

This is the design decision that shapes every number the tool reports.

**`--mode sequential`** reads every frame in order and never drops one. Every
model sees identical input, so throughput is reproducible. Use it to compare
models on a file.

**`--mode realtime`** runs the decoder in its own thread and keeps only the
newest frame. Whatever the consumer was too slow to process is dropped and
counted. This is what a live camera does to you.

Reading frames inline with inference instead, which is what most example code
does, lets the FFmpeg buffer absorb the backlog. Throughput looks fine while
latency grows without bound, and the run reports no dropped frames because none
were dropped, they were queued. The failure is real either way; realtime mode
makes it visible in the place you are looking.

A local file in realtime mode is throttled to its own frame rate by default, so
it behaves like a camera rather than racing through at disk speed. Override with
`--pace-fps`.

## Model catalogue

Licence is a first-class column, not a footnote. Run `visionbench models` for
the current list.

| Key | Model | Licence | Commercial use |
| --- | --- | --- | --- |
| `detr-r50`, `detr-r101` | DETR ResNet-50 / 101 | Apache-2.0 | yes |
| `rtdetrv2-r18`, `rtdetrv2-r50` | RT-DETRv2 | Apache-2.0 | yes |
| `dfine-nano`, `dfine-small`, `dfine-medium` | D-FINE | Apache-2.0 | yes |
| `rfdetr-nano`, `rfdetr-small`, `rfdetr-base` | RF-DETR | Apache-2.0 | yes |
| `yolo11n`, `yolo11s` | YOLO11 | AGPL-3.0 | no, without a paid licence |

The YOLO entries are installed only via the `agpl-baseline` extra and are never
a default. They are here because the YOLO family is the reference everyone
quotes, and a comparison that omits it invites the question. AGPL-3.0 obliges
you to publish the source of any networked service built on those weights, or
buy a commercial licence from the vendor. The RF-DETR Plus variants (XL and 2XL)
are under a separate proprietary licence and are deliberately absent from this
catalogue. See [docs/licensing.md](docs/licensing.md).

Adding a model that follows the Hugging Face `AutoModelForObjectDetection`
contract costs one registry entry and no code.

## First results

One machine so far. The sample clip is 768x576 at 10 fps, filtered to the
`person` class, fp32, Apple Silicon GPU through Metal.

| Model | Mode | FPS | Streams | Infer p50 | Infer p95 | E2E p95 | GPU mem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dfine-nano | sequential | 27.7 | 2.77 | 28.1 ms | 31.8 ms | 39.8 ms | 30 MB |
| dfine-small | sequential | 20.1 | 2.01 | 40.8 ms | 50.8 ms | 58.5 ms | 82 MB |
| rtdetrv2-r18 | sequential | 15.6 | 1.56 | 54.9 ms | 63.5 ms | 71.3 ms | 125 MB |
| dfine-nano | realtime | 10.0 | 1.00 | 35.7 ms | 39.9 ms | 47.7 ms | 20 MB |
| detr-r50 | sequential | 5.5 | 0.55 | 162.4 ms | 199.8 ms | 210.7 ms | 269 MB |

Host: Apple M1 Pro, 16 GB, `mps`. Reproduce with
`visionbench sweep -s clips/vtest.avi -m dfine-nano,dfine-small,rtdetrv2-r18,detr-r50 --mode sequential -d 20 -c person`.

Two things worth reading off this. D-FINE Nano is five times the throughput of
DETR ResNet-50 for a ninth of the GPU memory, which is the gap between one box
serving two cameras and one box not managing a single one. And the realtime row
is the same model against the same clip played at camera speed: it settles at
exactly the stream rate with nothing dropped, which is what headroom looks like.

Raw JSON for every run is under [`benchmarks/`](benchmarks), including host
fingerprint, per-stage percentiles and resource counters.

## Install

```bash
git clone https://github.com/frankville/ai-vision-detector-bench
cd ai-vision-detector-bench
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .

# optional backends
uv pip install -e '.[rfdetr]'          # RF-DETR
uv pip install -e '.[cuda]'            # NVIDIA telemetry via NVML
uv pip install -e '.[track]'           # multi-object tracking
uv pip install -e '.[agpl-baseline]'   # YOLO reference, read docs/licensing.md first
```

## Use

```bash
# what can this machine do, and which telemetry is available
visionbench doctor

# the catalogue, with licences
visionbench models

# one model against a clip
visionbench run -s clips/corridor.mp4 -m dfine-nano -d 30

# CUDA automatic mixed precision
visionbench run -s clips/corridor.mp4 --device cuda --backend pytorch --precision amp-fp16

# repeated measurements and aggregate statistics
visionbench sweep -s clips/corridor.mp4 --mode sequential --repeat 5
visionbench report --kind aggregate --full

# several models against a live camera, people only
visionbench sweep -s "rtsp://user:pass@host:554/stream" \
  -m dfine-nano -m rtdetrv2-r18 -m detr-r50 -c person -d 60

# comparison table across everything recorded so far
visionbench report
visionbench report --markdown

# dashboard: live view, live metrics, saved results
visionbench serve
```

The dashboard runs the same loop as the CLI and reads its progress from the same
meters, so what the browser shows cannot drift from what lands in the result
file. Annotating and encoding frames for the preview does cost CPU inside the
measured loop, so a watched run reports slightly lower throughput than the same
run headless. Runs started from the browser are tagged in their result file for
exactly that reason. Publish numbers from the CLI.

Results are written to `benchmarks/<host-profile>/` as JSON, one file per run,
and are meant to be committed. Comparing the same model across an Apple M-series
laptop, an older Intel box and a rented GPU instance is the point.

### What lands in a result file

Result files are designed to be committed to a public repository, so what goes
into them is a deliberate list rather than whatever was easy to collect.

Recorded: CPU model, core counts, RAM, OS and architecture, Python and PyTorch
versions, GPU name and memory, the profile label, and every measurement.

Not recorded: the machine's hostname. Hostnames routinely carry a person's name
or an employer's, and they say nothing about how fast a model ran. The `profile`
field identifies a machine instead, and you choose it with `--profile`.

Schemas v2 and v3 store requested settings in `config` and effective device, backend,
and precision in `model`. Reports retain v1 results but mark their precision as
unverified. PyTorch is currently the only inference backend.

Stream URLs are redacted wherever they appear, in files and in terminal output,
so a result carrying camera credentials cannot be committed by accident.

If you have older results collected before this was tightened:

```bash
python scripts/scrub_results.py
```

### Precision

`--precision fp32` uses FP32 weights and inputs. `fp16` uses FP16 weights and
floating pixel inputs on CUDA or MPS. `amp-fp16` keeps FP32 weights and inputs
and uses CUDA autocast around inference. Unsupported combinations fail.

Explicit FP16 casts floating pixel inputs on CPU before device transfer; masks
and integer metadata keep their dtypes. Casting and transfer remain part of
preprocessing timing. This differs from v1's cast after transfer, so compare
preprocessing results with that change in mind. Warmup and FPS definitions are
unchanged. E2E starts after decode; energy estimates cover the GPU only.

NVML currently samples physical GPU 0. Resource and energy figures may not match
the compute device with indexed CUDA selection or `CUDA_VISIBLE_DEVICES` remapping.

### Repeated runs

`sweep --repeat N` loads, warms up, measures, and unloads each attempt separately.
Models run serially, interleaved by repetition. The default remains one run.

Every raw result is saved before aggregation. Schema v3 adds run/group IDs,
repetition numbers, and completed, empty, or interrupted status. Writes are atomic
and never replace an existing result.

Repeated sweeps save separate summaries under `benchmarks/<profile>/aggregates/`.
Use `report --kind aggregate` or `report --kind aggregate --markdown` to read them;
ordinary reports and the dashboard continue to show raw runs.

Summaries report eligible/requested runs, FPS median/mean/range/sample standard
deviation, and medians of per-run latency percentiles. GPU counter summaries use
medians of per-run means; missing telemetry remains null with availability counts.
Sample standard deviation is null for fewer than two eligible runs.

Interrupted and empty runs remain saved but are excluded from statistics. A run
needs at least two completions and a positive completion interval. Partial groups
list failed, interrupted, empty, and unattempted slots. Repeated sweeps with failed
or empty attempts exit with code 1; interruption stops scheduling and exits 130.

Aggregates cover explicit groups from one sweep. Comparing arbitrary saved runs
or different hosts requires the workload/environment checks planned in the roadmap.

## What this does not do

Being explicit, because a benchmark that overstates itself is worse than none.

- **It does not measure accuracy.** Without labelled frames from your own
  cameras there is no precision or recall to report, only throughput and
  agreement. Detections per frame and mean confidence are descriptive, not a
  quality score. Labelled evaluation is on the roadmap below.
- **Utilisation on Apple Silicon is not available.** PyTorch reports the GPU
  memory it allocated; utilisation and power need `powermetrics` with root.
  Those fields are reported as null rather than estimated.
- **Estimated streams per host is an upper bound.** It divides processed fps by
  the stream rate and ignores the decode cost each extra camera adds.
- **One model at a time.** Two models on one device compete for the same silicon
  and both numbers come out wrong, so a sweep runs them in sequence.

## Roadmap

- [x] Threaded capture with explicit drop accounting
- [x] Stage-level timing and resource telemetry
- [x] DETR, RT-DETRv2, D-FINE, RF-DETR adapters
- [x] JSON results and comparison tables
- [x] FastAPI dashboard with an MJPEG view and live metrics
- [ ] Multi-object tracking and zone crossing counts
- [ ] Labelled evaluation set with precision, recall and mAP
- [ ] Cross-model agreement analysis, which needs no labels
- [ ] ONNX Runtime and OpenVINO backends for older CPUs

## Licence

Apache-2.0. See [LICENSE](LICENSE). Model weights carry their own licences,
listed above and reported by `visionbench models`.

## Development

The optional [agent workflow](docs/development/agent-workflow.md) defines project
roles and skills. The [benchmark roadmap](docs/development/roadmap.md) tracks
precision, reproducible GPU suites, and multicamera capacity work.

See [testing](docs/development/testing.md) for test layers and CI. Fast CPU tests
run on each PR update; the installed-CLI E2E workflow runs on demand.

```bash
uv pip install -e '.[dev]'
python -m pytest  # Unit and offline integration tests; E2E is opt-in
ruff check src tests
```
