# Dataset runbook — from camera footage to trained models

Every step is a `watchinu` command, re-runnable and recorded in the manifest.

```
footage ─► extract ─► dedupe ─► visit log ─► split ─► CVAT labelling ─► export ─► train ─► replay ─► eval
```

## 0. Rules that are never broken

1. **Never train on phone photos or web-UI screenshots** of the monitor (glare, moiré,
   scaling). Use recordings from the stream or exports from the NVR.
2. **Freeze the camera settings** (Video and Image tabs) before collecting. Any change to
   resolution, quality, WDR or exposure is a new data domain: write it down with the date.
3. **Split by visit, never by frame** (`watchinu split` enforces this).
4. **Label only what is visible.** Hidden coils are never boxed. The true count comes from
   the visit log.
5. **The gold test set is frozen** (`--frozen-test`) and is never used for tuning.

## 1. Get footage

**A. Record live** (preferred: bit-identical to what the runtime sees). Use a dedicated
read-only camera user, not the IT/admin account:

```bash
cp .env.example .env   # fill WATCHINU_CAM01_USER / WATCHINU_CAM01_PASSWORD
set -a; . ./.env; set +a
watchinu capture --config configs/camera01.yaml --stream main --out data/raw
# optional, for a sub-stream stage model later:
watchinu capture --config configs/camera01.yaml --stream sub --out data/raw
```

Expect about 66 GB per day for the main stream and 11 GB per day for the sub-stream at the
configured bitrates (VBR, usually less). Segments are 5-minute `.mkv` files named with their start time.

**B. NVR exports / the big recording.** Any `.mp4/.mkv/.avi`. If the file name does not
contain `YYYYMMDDTHHMMSS`, pass the wall-clock start time when extracting, otherwise
frames cannot be matched to the visit log.

## 2. Extract frames

```bash
watchinu extract --input data/raw/cam01/main --out data/frames --every-s 5
# single NVR export with no timestamp in its name:
watchinu extract --input export.mp4 --out data/frames --every-s 5 --start-time 2026-10-08T14:00:00
```

The command is resumable: already-extracted videos are skipped, and the manifest is saved after every video.
Each frame records its source video, presentation time, wall-clock time, blur, brightness,
over-exposure and dHash.

## 3. Remove near-duplicates

```bash
watchinu dedupe --manifest data/frames/manifest.jsonl --max-distance 6
```

Duplicates are only *marked* (`duplicate_of`), never deleted. Exports skip them by default.

## 4. Write the visit log (most valuable labelling hour you will spend)

```bash
watchinu visit-log-template --out data/visit_log.csv   # see configs/visit_log_template.csv
```

Write one row per truck: `arrived`, `loading_start`, `tarp_start`, `tarp_done`, `departed`,
`layout`, `coil_count`. Fill it from the footage, and cross-check `coil_count` against the
loading slip, dispatch note or weighbridge. From these timestamps the tools derive **a stage
label for every frame and a layout label for every open-bed frame**, so no per-frame tagging
is needed. The log must be complete for any footage used to train the stage model; an
unlogged truck would be labelled `no_truck`.

Ask the lacing team lead to log truck number, layout and count for two weeks. This is
independent ground truth for the inferred-count logic.

## 5. Split

```bash
watchinu split --manifest data/frames/manifest.jsonl --visit-log data/visit_log.csv \
  --val 0.15 --test 0.2 --seed 42 --frozen-test data/gold_test_visits.txt
```

Overlapping visits (two bays at once) stay together. Frames outside visits are grouped by
day. Write `gold_test_visits.txt` once, after the first split. From then on those visits stay in
test as the dataset grows.

## 6. Label boxes in CVAT (self-hosted; footage does not leave the plant network)

1. Create a CVAT task with labels `coil`, `person` and upload the non-duplicate frames
   (prioritise variety across time of day, weather, truck type and layout).
2. **Pre-label** so annotators correct instead of draw:
   ```bash
   # zero-shot bootstrap (open-vocabulary model)
   watchinu prelabel --weights yolov8s-worldv2.pt --images data/frames/images \
     --prompts "wrapped wire coil=coil,person=person" --out data/prelabel_v0.zip
   # once ~200 frames are labelled and a v0 detector is trained:
   watchinu prelabel --weights runs/detect/coil-v0/weights/best.pt --images ... --out ...
   ```
   Import the zip in CVAT as **YOLO 1.1**.
3. Follow `docs/labelling_guide.md`. Double-label 10–15% of frames to measure agreement.
4. Export as **YOLO 1.1** and unzip the `.txt` files into `data/labels/`.

## 7. Export training datasets (cropped exactly like the runtime)

```bash
watchinu export-detect --manifest data/frames/manifest.jsonl --labels data/labels \
  --out data/ds/detect --classes coil,person --config configs/camera01.yaml --bay left
watchinu export-cls --manifest data/frames/manifest.jsonl --visit-log data/visit_log.csv \
  --task stage --out data/ds/stage --config configs/camera01.yaml --bay left
watchinu export-cls --manifest data/frames/manifest.jsonl --visit-log data/visit_log.csv \
  --task layout --out data/ds/layout --config configs/camera01.yaml --bay left
```

Each output has a `dataset_card.json` (counts per split and class, plus a manifest digest).

## 8. Train (GPU machine)

```bash
pip install -e ".[ml]"
watchinu train --data data/ds/detect/data.yaml --model yolo26s.pt --imgsz 1280 \
  --hyp configs/train/detector.yaml --name coil-v1
watchinu train --data data/ds/stage --model yolo26n-cls.pt --imgsz 224 \
  --hyp configs/train/classifier.yaml --name stage-v1
watchinu train --data data/ds/layout --model yolo26s-cls.pt --imgsz 384 \
  --hyp configs/train/classifier.yaml --name layout-v1
```

Every run writes `provenance.json` (git commit, dataset card, args).

## 9. Replay and evaluate per visit

```bash
# point configs/camera01.yaml models.* at the new weights, then:
watchinu replay --config configs/camera01.yaml --video data/raw/cam01/main --out runs/replay/v1
watchinu eval --predictions runs/replay/v1/events.jsonl --visit-log data/visit_log.csv \
  --out runs/replay/v1/report.md
# isolate the heavy tier from visit detection errors:
watchinu replay ... --stage-oracle data/visit_log.csv
```

The report gives visit recall, layout accuracy and confusion, count exact-match rate and MAE,
the observed ≤ true rate, and the review rate. Report every metric **on the gold test visits** and per slice
(day/night, truck type).

## Coverage checklist

| Dimension | Cover deliberately |
|---|---|
| Light | morning, harsh midday sun, evening, night |
| Weather | sun, overcast, rain, dust |
| Truck | wooden-sided, steel-sided, trailer; own and vendor/customer |
| Load | linear, criss-cross, mixed, partial, dark-wrapped coils |
| Scene | many workers, crane or forklift, both bays occupied |
| Negatives | empty bay, truck passing through, covered truck, workers without truck |

Criss-cross loads are probably the rare class. Ask supervisors for dates and shifts and pull those
visits from the NVR on purpose.

Rough targets: 150–300 visits, 2,000–3,000 labelled detector frames, and at least 10–15
gold-test visits per layout class. Revise them from the learning curve.
