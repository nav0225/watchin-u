# watchin-u — wire-rod coil load monitoring

Computer-vision monitoring of wire-rod coil loading at the **lacing point, Material Gate
(Parsabhata Road), BALCO**. From the Hikvision camera over the lacing bay it decides, for
every truck and before the tarpaulin hides the load:

* the **product**: wrapped wire rods, or something else (ingots, ...) that is logged and ignored;
* the **arrangement**: linear (2×5 = 10) / criss-cross (~2×7 = 14) / mixed;
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
| Storage: SQL Server / PostgreSQL / SQLite schema, migrations, append-only decisions, outbox, `db-check` | built; tested on SQLite, SQL Server DDL verified offline |
| Dataset tooling: capture → extract → dedupe → visit log → split → export → pre-label | built and tested |
| Models | **not trained yet**: Colab notebooks ready (`notebooks/`); bootstrap pre-labeller works on real footage |
| Deployment | Windows/Linux service, SQL Server / PostgreSQL ([docs/deployment.md](docs/deployment.md)) |
| Geometry in `configs/camera01.yaml` | calibrated on real footage: two lanes (`far`, `near`), see [data report 2026-11-04](docs/data_reports/2026-11-04.md) |
| Labels | draft per-lane visit log for 2026-11-04 in `labels/`, pre-labelled CVAT bundle to correct |
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
| `capture` | Record the camera: lossless 5-min segments, or `--mode timelapse` JPEGs (auto-reconnect) |
| `extract` | Sample frames from videos / index time-lapse JPEGs into `manifest.jsonl` (resumable) |
| `visit-log-from-clips` | One pre-filled visit-log row per short clip |
| `contact-sheet` | Thumbnail grids per clip for quick review |
| `dedupe` | Mark near-duplicate frames |
| `visit-log-template` | Empty visit-log CSV (per-truck ground truth) |
| `split` | Leakage-safe train/val/test split by visit, with a frozen gold test set |
| `export-detect` / `export-cls` | Ultralytics datasets cropped exactly like the runtime |
| `prelabel` | Model-assisted labelling: CVAT "YOLO 1.1" import zip |
| `train` | Fine-tune with provenance (`provenance.json` per run) |
| `replay` | Full pipeline on recorded footage (optionally with ground-truth stages) |
| `eval` | Visit-level metrics vs the visit log (product, layout, count) |
| `evaluate-model` | Score trained weights on the gold test split (mAP, confusion matrix) |
| `db-check` | Smoke-test the database: connect, schema version, write/read round trip |
| `run` | Live on the camera, writing to the database |
| `render-config` | Draw bays, bed quads, slots and OSD masks on a frame to check calibration |
| `db-upgrade` | Apply Alembic migrations |

## Security

Camera credentials come only from the environment (`.env` is git-ignored). Use a
dedicated read-only camera user, never the admin/IT account. URLs are redacted in logs.
Footage, frames, weights and databases never go into git.
