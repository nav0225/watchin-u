# Dataset runbook — from camera footage to trained models

Every step is a `watchinu` command, re-runnable and recorded in the manifest.

**Easiest path (no Python on your laptop):** put the footage zip in Google Drive and run
`notebooks/01_prepare_dataset_colab.ipynb` (frames, visit log, contact sheets, CVAT
pre-labels, and later the datasets), then `notebooks/02_train_colab.ipynb` (train, evaluate,
summary to paste back). The commands below are what those notebooks run.

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

**B. Time-lapse of whole visits** (best value per GB). One JPEG every 10 s covers a
40-minute visit in 240 frames, at about 1/7 of the video storage (measured: ~1 MB per
frame, ~0.36 GB/hour, versus ~2.5 GB/hour of video):

```bash
watchinu capture --config configs/camera01.yaml --mode timelapse --every-s 10 --out data/raw
```

**C. Short clips / NVR exports / phone-free downloads.** Any `.mp4/.mkv/.avi`. If the file name
does not contain `YYYYMMDDTHHMMSS`, pass `--start-time` (one video) or
`--start-time-from-mtime` (clips; uses the file time). Ten-second clips are fine for the
**detector and layout** models: variety across trucks matters more than length. The
**stage** model and the trigger timing need some whole visits (time-lapse), plus clips of the
empty bay and of the tarp going on.

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
watchinu dedupe --manifest data/frames/manifest.jsonl --config configs/camera01.yaml
```

Rule, per video against the last kept frame: keep if the scene changed (largest per-region
change at least 0.03) and at least 3 s have passed, or after 10 s regardless. Static clips give a
frame every 10 s, and busy ones (tarping, lacing) give more. The ticking OSD clock is blanked
first. On 2026-11-04 this kept 86 of 393 frames. A whole-frame perceptual hash had
collapsed 365 of them, including busy clips. Duplicates are only *marked*, never deleted.

## 4. Write the visit log (most valuable labelling hour you will spend)

```bash
watchinu visit-log-template --out data/visit_log.csv   # see configs/visit_log_template.csv
# short clips: one pre-filled row per clip; you fill product / layout / coil_count
watchinu visit-log-from-clips --manifest data/frames/manifest.jsonl --out data/visit_log.csv
watchinu contact-sheet --manifest data/frames/manifest.jsonl --out data/contact_sheets
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

**Rows are per lane:** one row per truck per lane (`bay_id` `far` / `near`). For partial
recordings use `window=clip`, and `stage` if it is constant over the recording. Give one
physical truck the same `truck_ref` in every row; such rows always share a split.

```bash
watchinu split --manifest data/frames/manifest.jsonl --visit-log data/visit_log.csv \
  --val 0.15 --test 0.2 --seed 42 --frozen-test data/gold_test_visits.txt
```

Overlapping visits (two bays at once) and rows of the same truck stay together. Frames
outside visits are grouped by day. On a day where the lanes are busy back-to-back this makes
the whole day **one group**, which is correct, and it can't be split. To bootstrap a first
model from such a single session, use `--group-by clip --test 0`. Each recording stays
whole, but one truck can then appear in train and val, so val scores are optimistic. Take
the gold test set from other days. Write `gold_test_visits.txt` once, after the first split. From then on those visits stay in
test as the dataset grows.

## 6. Label boxes in CVAT (self-hosted; footage does not leave the plant network)

1. Create a CVAT task with labels `coil`, `person` and upload the non-duplicate frames
   (prioritise variety across time of day, weather, truck type and layout).
2. **Pre-label** so annotators correct instead of draw:
   ```bash
   # no labels yet: YOLOE prompt-free proposals + the BALCO wrap colour test (+ COCO persons)
   watchinu prelabel --method wrap-colour --manifest data/frames/manifest.jsonl \
     --out data/cvat/prelabel.zip --images-zip data/cvat/images.zip --device 0
   # once ~200 frames are corrected and a v0 detector is trained:
   watchinu prelabel --method weights --weights runs/detect-v1/weights/best.pt --manifest ...
   ```
   Create the CVAT task from `images.zip`, then import `prelabel.zip` as **YOLO 1.1**.
   Text-prompted open-vocabulary models (`--prompts`) need a CLIP install. Don't use them
   on this footage: they named the coils "sleeping bag" and "laundry". The wrap colour test
   is what separates coils from ingots and tarps.
3. Follow `docs/labelling_guide.md`. Double-label 10–15% of frames to measure agreement.
4. Export as **YOLO 1.1** and unzip the `.txt` files into `data/labels/`.

## 6b. Audit the labels before every training run

```bash
watchinu label-audit --manifest data/frames/manifest.jsonl --labels data/labels \
  --visit-log data/visit_log.csv --config configs/camera01.yaml
```

The audit reports coils per lane and recording next to the visit log, and flags coils in an
empty lane, coils on a non-wire-rod or covered truck, and more coils than `coil_count`.
Run it on pre-labels too, to judge how much correcting they need.

## 7. Export training datasets (cropped exactly like the runtime)

```bash
# detector: full frames, OSD blanked (exactly what the runtime detector sees)
watchinu export-detect --manifest data/frames/manifest.jsonl --labels data/labels \
  --out data/ds/detect --classes coil,person --config configs/camera01.yaml
# classifiers: one crop per lane, labelled from that lane's visits
watchinu export-cls --manifest data/frames/manifest.jsonl --visit-log data/visit_log.csv \
  --task stage --out data/ds/stage --config configs/camera01.yaml
watchinu export-cls --manifest data/frames/manifest.jsonl --visit-log data/visit_log.csv \
  --task layout --out data/ds/layout --config configs/camera01.yaml
```

Each output has a `dataset_card.json` (counts per split and class, plus a manifest digest).

## 8. Train (GPU machine; Colab: `notebooks/02_train_colab.ipynb`)

```bash
pip install -e ".[ml]"
watchinu train --data data/ds/detect/data.yaml --model yolo26s.pt --imgsz 1280 \
  --hyp configs/train/detector.yaml --name coil-v1
watchinu train --data data/ds/stage --model yolo26n-cls.pt --imgsz 224 \
  --hyp configs/train/classifier.yaml --name stage-v1
watchinu train --data data/ds/layout --model yolo26s-cls.pt --imgsz 384 \
  --hyp configs/train/classifier.yaml --name layout-v1
```

Every run writes `provenance.json` (git commit, dataset card, args). Score each on the frozen
test split, and paste the markdown back:

```bash
watchinu evaluate-model --task detect --weights runs/coil-v1/weights/best.pt --data data/ds/detect/data.yaml
watchinu evaluate-model --task classify --weights runs/stage-v1/weights/best.pt --data data/ds/stage --imgsz 224
```

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
