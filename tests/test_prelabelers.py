from __future__ import annotations

import numpy as np

from watchinu.domain import BBox, Detection
from watchinu.models.fake import FunctionDetector
from watchinu.models.prelabelers import CompositeDetector, LabelFilter, select_coils

W, H = 400, 300
WHITE, BLUE, GREY, YELLOW = (225, 225, 225), (200, 60, 20), (160, 160, 160), (0, 200, 230)


def patch(img, box, colour_top, colour_bottom=None):
    x1, y1, x2, y2 = (int(v) for v in (box.x1, box.y1, box.x2, box.y2))
    mid = (y1 + y2) // 2
    img[y1:mid, x1:x2] = colour_top
    img[mid:y2, x1:x2] = colour_bottom or colour_top
    mask = np.zeros(img.shape[:2], bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_select_coils_keeps_wrapped_coils_and_rejects_everything_else():
    img = np.full((H, W, 3), 40, np.uint8)
    coil_a, coil_b = BBox(20, 20, 100, 100), BBox(110, 20, 190, 100)
    ingot, tarp = BBox(220, 20, 300, 100), BBox(20, 150, 120, 250)
    boxes, confs, masks = [], [], []
    for box, conf, top, bottom in (
        (coil_a, 0.6, WHITE, BLUE),
        (coil_b, 0.5, WHITE, BLUE),
        (ingot, 0.7, GREY, GREY),  # grey metal: "white-ish" but no blue band
        (tarp, 0.8, YELLOW, YELLOW),
    ):
        masks.append(patch(img, box, top, bottom))
        boxes.append(box)
        confs.append(conf)
    # a group box over both coils, a duplicate of coil A, and a tiny speck
    group = BBox(15, 15, 195, 105)
    group_mask = masks[0] | masks[1]
    duplicate = BBox(22, 22, 98, 98)
    speck = BBox(300, 200, 305, 205)
    speck_mask = patch(img, speck, WHITE, BLUE)
    boxes += [group, duplicate, speck]
    confs += [0.9, 0.3, 0.9]
    masks += [group_mask, masks[0], speck_mask]

    out = select_coils(img, boxes, confs, masks)
    assert sorted((d.box.x1, d.box.y1) for d in out) == [(20, 20), (110, 20)]
    assert all(d.label == "coil" for d in out)


def test_label_filter_and_composite():
    coco = FunctionDetector(
        lambda img: [
            Detection("person", 0.9, BBox(0, 0, 10, 10)),
            Detection("truck", 0.9, BBox(0, 0, 50, 50)),
        ],
        name="coco",
    )
    coils = FunctionDetector(lambda img: [Detection("coil", 0.8, BBox(5, 5, 20, 20))], name="coils")
    both = CompositeDetector(coils, LabelFilter(coco, {"person": "person"}))
    labels = [d.label for d in both.detect(np.zeros((10, 10, 3), np.uint8))]
    assert labels == ["coil", "person"]
    assert both.name == "coils+coco"
