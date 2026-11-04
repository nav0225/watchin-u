# Labelling guide

Consistency matters more than speed. When unsure, use the rule here; if the rule does not
cover it, add an example to this guide so everyone labels it the same way.

## Detector boxes (CVAT, label set `coil`, `person`)

### `coil`: one wrapped wire-rod coil (package)

* Box **every coil whose visible part is ≥ 30% of the coil** (white/blue Vedanta/BALCO wrap
  *and* dark-wrapped coils). Draw the box tight around the **visible** part only.
* **Do not** box coils that are fully hidden, even if you "know" they are there. Hidden
  coils are counted through the visit log, not boxes.
* Box coils on **both** trucks if both are visible. The bay polygon separates them at
  export and runtime, so do not crop mentally.
* Coils hanging from the crane or on the forklift: box them (they are coils). Placement is
  handled by slot logic.
* Straps or lacing over a coil: box the coil as if the straps were not there.
* Two coils that look merged: two boxes, split at the visible seam. If no seam is visible at all,
  one box plus an `ambiguous` CVAT tag on the frame.

### `person`

* Box every worker whose body is ≥ 30% visible, including people standing on the truck walls.
  These boxes drive the occlusion logic.

### Frame-level

* A frame with nothing to box (empty bay, covered truck) is still **saved as a labelled
  frame with no boxes**. It is a verified negative, and the detector needs these.
* Frames that are unusable (blank, frozen, corrupted) get the CVAT tag `unusable` and are skipped.

## Visit log (one row per truck)

| Field | Definition |
|---|---|
| `arrived` | Truck stops in the bay (wheels stationary). |
| `loading_start` | First coil placed on the bed. **Required for layout training.** |
| `tarp_start` | Tarp first drawn over any coil. |
| `tarp_done` | Bed fully covered. |
| `departed` | Truck starts leaving the bay. |
| `product` | `wire_rod` for wrapped wire-rod coils. Anything else in plain words: `ingot`, `empty`, ... Every non-wire-rod value trains the `other_load` class. Leave it blank only if unknown (that visit's open-bed frames are then skipped). |
| `layout` | `linear` (2 columns × 5, coils standing in rows), `criss_cross` (~2 × 7), or `mixed` (both in one truck). Blank for non-wire-rod trucks. |
| `coil_count` | Final number of coils on the truck. Take it from the loading slip, dispatch note or weighbridge, not from the camera. |
| `bay_id` | Lane: `far` (along the shed wall) or `near` (road side). **One row per lane**: two trucks at once means two rows. |
| `truck_ref` | Same value for one physical truck seen in several recordings (for example `2026-11-04/V3`). Such rows always share a split. |
| `window` | `clip` when the row describes a short recording rather than a whole visit (written by `visit-log-from-clips`). Arrival and departure are then just the recording bounds. |
| `stage` | Optional. What was visible for the whole window when the real tarp times were not observed: `wire_rod`, `other_load`, `covering`, `covered`. It overrides the tarp columns. |

Stage labels per frame follow automatically: `no_truck` outside visits; `other_load` for
non-wire-rod trucks, and for a wire-rod truck before `loading_start` (empty bed); `wire_rod`
while coils are visible; `covering` from `tarp_start`; `covered` from `tarp_done`.

Times are plant local time (IST) unless an offset is written. Use the OSD clock on the
footage. If the camera clock is off, note the offset once in `notes`.

## Agreement check

Double-label 10–15% of frames. Disagreements on "is this 30% visible?" or merged coils go
into this guide as new examples, with a frame id.
