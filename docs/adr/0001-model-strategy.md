# ADR 0001 — Model strategy: fine-tune pretrained models, never train from scratch

- **Status:** accepted
- **Date:** 2026-10-31

## Context

We must recognise, from one fixed Hikvision camera at the BALCO lacing point:

1. the **visit stage** of a truck (no truck / wrapped wire rods / other load such as ingots /
   tarp going on / covered). The heavy model only ever wakes for wire-rod trucks;
2. the **coil arrangement** (linear 2×5 = 10 vs criss-cross ~2×7 = 14 vs mixed);
3. the **visible coil count** and which template slots are occupied.

No public model detects wrapped wire-rod coils. The footage is a narrow domain: one
viewpoint, one product family, two wrap colours. We expect a few hundred labelled visits
and a few thousand labelled frames in the delivery window. Legal has cleared AGPL /
Enterprise licensing, so licence is not a constraint.

## Decision

**Fine-tune COCO/ImageNet-pretrained models on our own footage.** Three small models,
each with one job:

| Role | Base model | Input | Why |
|---|---|---|---|
| Stage classifier (light tier, runs every 5–10 s) | `yolo26n-cls` (ImageNet) | bay crop, 224 px | Tiny and cheap. Classes `no_truck`, `wire_rod`, `other_load`, `covering`, `covered`. Wrapped coils and ingots look very different. |
| Layout classifier (heavy tier) | `yolo26s-cls` (ImageNet) | bay crop, 384 px | Arrangement is a whole-image pattern. |
| Coil + person detector (heavy tier) | `yolo26s` (COCO) | bay crop, 1280 px | `person` is already learned. Coils are large, few objects. |

* **Framework:** Ultralytics YOLO26 (`ultralytics` 8.4.x). It has the most mature tooling for
  fine-tuning, validation, and export to ONNX, TensorRT and OpenVINO. YOLO26 inference is end-to-end
  (NMS-free), which simplifies deployment.
* **Challenger:** RF-DETR (Apache-2.0, DINOv2 backbone). At milestone M2 we train it on the
  same dataset and compare on the frozen gold set at equal latency. The runtime depends
  only on the `Detector` / `Classifier` protocols (`src/watchinu/models/base.py`), so
  switching is a small adapter, not a rewrite.
* **Labelling bootstrap:** open-vocabulary detection (YOLO-World / YOLOE prompts such as
  "wrapped wire coil") via `watchinu prelabel` produces first-pass boxes for CVAT. Once
  about 200 frames are corrected, our own v0 detector takes over pre-labelling (model-assisted
  loop).
* **Counting:** count from detections plus slot geometry. Density-map counting is overkill
  for ~10 large objects.

## Alternatives rejected

| Option | Why not |
|---|---|
| Train from scratch | Needs 10–100× more labelled data for the same accuracy. Gains nothing on a narrow domain. |
| Zero-shot foundation models at runtime (Grounding DINO, SAM 2, VLMs) | Too heavy for 24/7 duty and not reliable enough to count wrapped coils. Used only for pre-labelling. |
| One end-to-end model for everything | Couples stage, layout and count failures. Prevents duty-cycling (the stage model must be cheap; the detector need not run often). |
| Multi-view 3D counting | Needs more cameras. Revisit in phase 2 if hidden-slot inference proves insufficient. |

## Consequences

* Accuracy depends on **our data**: coverage of night, rain, glare, truck types and both layouts
  (see `docs/dataset_runbook.md`). The visit log is the most valuable labelling artefact.
* Every trained weight file carries `provenance.json` (git commit, dataset digest, args),
  so any production result can be traced to the exact code and data.
* Deployment target: ONNX Runtime / TensorRT FP16 on the inference server, or OpenVINO on
  CPU. The heavy tier runs about 30 times per 45-minute visit, so a single modest GPU or even a CPU suffices.
