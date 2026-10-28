# Testing

## Layers

| Layer | Coverage | Execution |
| --- | --- | --- |
| Unit | Configuration, precision control flow, repetition scheduling, statistics, timer boundaries, failure injection | Every PR update |
| Integration | Typer/FastAPI components, filesystem publication, real OpenCV decoding, runner/source/report pipeline, local random DETR inference | Every PR update, offline on CPU |
| E2E | Installed wheel, real CPU D-FINE inference, generated clip, CLI run/sweep, saved JSON, raw/aggregate reports | Manual workflow dispatch |

Unmarked tests default to `unit`; each test has exactly one layer. Component tests may substitute a detector or runner and are not E2E. The real HF integration generates a tiny random checkpoint locally; it checks execution contracts, not accuracy.

The original suite covered mocked adapter/runner paths, filesystem persistence, aggregation, and in-process CLI/API calls. It lacked real decode and HF inference coverage, installed-CLI tests, category markers, and CI workflows. New integrations cover these boundaries and known timing definitions.

There is no mutation test suite. Changing inputs or injecting exceptions is fault injection, not mutation testing. A later optional mutation job can target pure configuration, timing, and aggregation logic after the CI baseline is stable. Browser flows, RTSP network behavior, GPU precision equivalence, and performance thresholds remain outside this CPU suite.

## Local commands

```bash
uv pip install -e '.[dev]'
python -m pytest                    # Unit and integration; E2E skipped
python -m pytest -m unit
python -m pytest -m integration
ruff check src tests
```

Fast tests set HF offline flags and block IP socket connections. Clips and random weights are generated in temporary directories. No credentials or GPU are required.

E2E requires an installed package and the registered D-FINE Nano checkpoint in the HF cache:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('ustc-community/dfine-nano-coco', allow_patterns=['*.json', 'model.safetensors'])"
python -m pytest tests/e2e -m e2e --run-e2e
```

Model staging may download public weights. The test itself runs offline, forces CPU/FP32, uses subprocess timeouts, and has no accuracy or FPS thresholds. CI builds and installs a wheel and checks that subprocesses import it outside the checkout. The resolved checkpoint snapshot appears in the workflow log; this is an execution smoke test, not a canonical hardware comparison.

## GitHub Actions

`Tests` runs lint and fast tests on Python 3.10 and 3.12 when a PR opens or receives commits, and after changes reach `main`. Superseded runs are cancelled. CPU wheels avoid installing CUDA dependencies.

`CPU end-to-end` uses only `workflow_dispatch`. After the workflow reaches the default branch, select it under Actions, choose a branch, and click **Run workflow**, or run:

```bash
gh workflow run e2e.yml --ref feat/repeated-benchmarks
```

GitHub requires the dispatch workflow to exist on the default branch before manual runs are available. See [manual workflow runs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).

Both workflows use read-only repository permissions and pinned action commits. Artifacts retain JUnit reports and E2E result JSON/logs for seven days; clips and model weights are excluded. Dependency locking remains part of LAB.3.

## Input-size regression

Real inference exposed ignored `--imgsz` overrides with Transformers processors using structured size objects. The adapter now accepts their dictionary protocol. Existing JSON remains unchanged; use effective `model.input_shape` when interpreting older runs and rerun affected size comparisons.
