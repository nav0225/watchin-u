# watchin-u — wire-rod coil load monitoring

Computer-vision monitoring of wire-rod coil loading at the **lacing point, Material Gate
(Parsabhata Road), BALCO**. From the Hikvision camera over the lacing bay it decides, for
every truck and before the tarpaulin hides the load:

* the **arrangement**: linear (2×5) / criss-cross / mixed;
* the **coil count**, with *observed* and *inferred* counts kept separate;
* whether a person must **review** it, and why;
* with an **evidence image** and the model versions used.

It does not run a heavy model 24/7. A cheap stage classifier watches the bay. A visit state
machine opens a visit when a truck stops, and the detector wakes only a few dozen times per
45-minute visit: around minute 2 for a provisional answer, after each loading burst, and over the last
3 minutes before the tarp for the final answer.

## Status

| Area | State |
|---|---|
| Runtime: ingest, state machine, scheduler, slots, decisions, drift guard | built and tested on synthetic footage |
| Storage: PostgreSQL/SQLite schema, migrations, append-only decisions, outbox | built and tested |
| Dataset tooling: capture → extract → dedupe → visit log → split → export → pre-label | built and tested |
| Models | **not trained yet**: waiting for footage and labels (see the runbook) |
| Geometry in `configs/camera01.yaml` | **placeholder**: calibrate on a real main-stream frame |
| ERP / weighbridge integration, live alerts | phase 2 |

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # core + tests
pip install -e ".[ml]"           # + Ultralytics/PyTorch (training and inference machines)
make check                       # ruff + mypy + pytest

watchinu render-config --config configs/camera01.yaml --image frame.jpg --out check.png
watchinu db-upgrade              # creates/migrates the database (WATCHINU_DATABASE_URL)
```

* Dataset and training: **[docs/dataset_runbook.md](docs/dataset_runbook.md)**
* Labelling rules: [docs/labelling_guide.md](docs/labelling_guide.md)
* How it works: [docs/architecture.md](docs/architecture.md)
* Why fine-tuned YOLO26 (and not from scratch): [docs/adr/0001-model-strategy.md](docs/adr/0001-model-strategy.md)

## Commands

| Command | Purpose |
|---|---|
| `capture` | Record the camera losslessly into 5-min time-stamped segments (auto-reconnect) |
| `extract` | Sample frames from footage into `manifest.jsonl` (resumable) |
| `dedupe` | Mark near-duplicate frames |
| `visit-log-template` | Empty visit-log CSV (per-truck ground truth) |
| `split` | Leakage-safe train/val/test split by visit, with a frozen gold test set |
| `export-detect` / `export-cls` | Ultralytics datasets cropped exactly like the runtime |
| `prelabel` | Model-assisted labelling: CVAT "YOLO 1.1" import zip |
| `train` | Fine-tune with provenance (`provenance.json` per run) |
| `replay` | Full pipeline on recorded footage (optionally with ground-truth stages) |
| `eval` | Visit-level metrics vs the visit log |
| `run` | Live on the camera, writing to the database |
| `render-config` | Draw bays, bed quads, slots and OSD masks on a frame to check calibration |
| `db-upgrade` | Apply Alembic migrations |

## Security

Camera credentials come only from the environment (`.env` is git-ignored). Use a
dedicated read-only camera user, never the admin/IT account. URLs are redacted in logs.
Footage, frames, weights and databases never go into git.
