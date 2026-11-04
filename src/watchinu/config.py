"""Typed configuration loaded from YAML. Secrets come only from environment variables.

Polygon coordinates are *normalised* (0..1 of frame width/height) so one config serves
both the main stream (2560x1440) and the sub-stream (640x360).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Point = tuple[float, float]
NormPolygon = Annotated[list[Point], Field(min_length=3)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_normalised(points: list[Point], lo: float = 0.0, hi: float = 1.0) -> list[Point]:
    for x, y in points:
        if not (lo <= x <= hi and lo <= y <= hi):
            raise ValueError(f"point {(x, y)} is outside {lo}..{hi} (normalised coordinates)")
    return points


class CredentialsEnv(_Strict):
    user: str
    password: str


class CameraConfig(_Strict):
    id: str
    name: str = ""
    host: str
    rtsp_port: int = 554
    streams: dict[str, str] = Field(
        default_factory=lambda: {
            "main": "/Streaming/Channels/101",
            "sub": "/Streaming/Channels/102",
        }
    )
    credentials_env: CredentialsEnv
    timezone: str = "Asia/Kolkata"
    # Burned-in OSD (timestamp, camera name) is blanked before any model sees the frame.
    osd_masks: list[NormPolygon] = Field(default_factory=list)
    # Region of static structure (shed trusses, columns) used by the view-drift guard.
    static_region: NormPolygon | None = None
    reference_image: str | None = None

    @field_validator("osd_masks")
    @classmethod
    def _osd_norm(cls, v: list[list[Point]]) -> list[list[Point]]:
        for poly in v:
            _check_normalised(poly)
        return v

    def credentials(self) -> tuple[str, str]:
        user = os.environ.get(self.credentials_env.user)
        password = os.environ.get(self.credentials_env.password)
        if not user or not password:
            raise RuntimeError(
                f"camera credentials missing: set {self.credentials_env.user} and "
                f"{self.credentials_env.password} in the environment"
            )
        return user, password


class SlotTemplate(_Strict):
    columns: int = Field(ge=1)
    rows: int = Field(ge=1)

    @property
    def n_slots(self) -> int:
        return self.columns * self.rows


class BayConfig(_Strict):
    id: str
    polygon: NormPolygon
    # Four corners of the bed plane at coil mid-height, in this order: near-left,
    # near-right, far-right, far-left (as seen from the camera). Corners may lie slightly
    # outside the frame when the truck's end is cut off by the image edge.
    bed_quad: Annotated[list[Point], Field(min_length=4, max_length=4)]
    # Layout -> slot template, e.g. {linear: std_2x5, criss_cross: cc_2x7}. A layout with
    # no template (e.g. mixed) falls back to a robust per-frame detection count.
    slot_templates: dict[str, str] = Field(min_length=1)

    @field_validator("slot_templates")
    @classmethod
    def _layouts(cls, v: dict[str, str]) -> dict[str, str]:
        allowed = {"linear", "criss_cross", "mixed"}
        unknown = set(v) - allowed
        if unknown:
            raise ValueError(f"unknown layout(s) {sorted(unknown)}; use {sorted(allowed)}")
        return v

    @field_validator("polygon")
    @classmethod
    def _norm(cls, v: list[Point]) -> list[Point]:
        return _check_normalised(v)

    @field_validator("bed_quad")
    @classmethod
    def _quad_norm(cls, v: list[Point]) -> list[Point]:
        return _check_normalised(v, -0.5, 1.5)


class VisitConfig(_Strict):
    open_after_s: float = 30.0
    close_after_s: float = 120.0
    covered_after_s: float = 60.0
    reopen_after_s: float = 120.0
    provisional_after_s: float = 120.0
    max_visit_s: float = 3 * 3600.0
    max_gap_s: float = 60.0
    stage_min_confidence: float = 0.6


class ScheduleConfig(_Strict):
    # Light tier (stage classifier + change score) cadence.
    idle_interval_s: float = 5.0  # no visit: catch arrivals quickly
    open_interval_s: float = 10.0  # visit open: also the look-back buffer cadence
    # Heavy tier (detector + layout classifier) triggers while a visit is open.
    provisional_interval_s: float = 10.0  # during the first ``provisional_after_s``
    checkpoint_interval_s: float = 600.0  # periodic re-check
    change_spike: float = 0.10  # loading burst started ...
    change_settle: float = 0.03  # ... and settled -> run heavy once
    lookback_s: float = 180.0  # frames kept to decide "just before the tarp"
    drift_check_interval_s: float = 300.0


class QualityConfig(_Strict):
    min_blur_var: float = 40.0
    min_brightness: float = 25.0
    max_overexposed_fraction: float = 0.5  # white wraps in sun; calibrate
    max_occlusion_fraction: float = 0.4


class DecisionConfig(_Strict):
    min_layout_votes: int = 5
    min_layout_agreement: float = 0.7
    min_layout_confidence: float = 0.6
    final_window_s: float = 900.0
    infer_unseen_slots: bool = True
    slot_history: int = 3
    coil_min_confidence: float = 0.4
    # Product gate: the heavy model wakes only after this many light-tier frames have
    # confirmed wrapped wire rods on the bed (at 10 s cadence: ~30 s of confirmation).
    min_wire_rod_votes: int = 3
    # Wire-rod share of open-bed frames within +/- this band of 0.5 -> product uncertain.
    product_uncertain_band: float = 0.2


class DriftConfig(_Strict):
    min_inlier_ratio: float = 0.3
    max_shift_px: float = 25.0


class ModelsConfig(_Strict):
    stage_classifier: str | None = None
    layout_classifier: str | None = None
    detector: str | None = None
    detector_imgsz: int = 1280
    stage_imgsz: int = 224  # coarse: is there a truck / a tarp?
    layout_imgsz: int = 384  # needs enough detail to see the arrangement
    device: str = "cpu"
    coil_label: str = "coil"
    person_label: str = "person"


class StorageConfig(_Strict):
    database_url_env: str = "WATCHINU_DATABASE_URL"
    default_database_url: str = "sqlite:///data/watchinu.db"
    evidence_dir: str = "data/evidence"

    def database_url(self) -> str:
        return os.environ.get(self.database_url_env, self.default_database_url)


class AppConfig(_Strict):
    camera: CameraConfig
    slot_templates: dict[str, SlotTemplate]
    bays: list[BayConfig] = Field(min_length=1)
    visit: VisitConfig = VisitConfig()
    schedule: ScheduleConfig = ScheduleConfig()
    quality: QualityConfig = QualityConfig()
    decision: DecisionConfig = DecisionConfig()
    drift: DriftConfig = DriftConfig()
    models: ModelsConfig = ModelsConfig()
    storage: StorageConfig = StorageConfig()

    @model_validator(mode="after")
    def _templates_exist(self) -> AppConfig:
        ids = [b.id for b in self.bays]
        if len(ids) != len(set(ids)):
            raise ValueError("bay ids must be unique")
        for bay in self.bays:
            for layout, name in bay.slot_templates.items():
                if name not in self.slot_templates:
                    raise ValueError(
                        f"bay {bay.id!r} uses unknown slot template {name!r} for {layout}"
                    )
        return self

    def bay(self, bay_id: str) -> BayConfig:
        for bay in self.bays:
            if bay.id == bay_id:
                return bay
        raise KeyError(f"unknown bay {bay_id!r}")

    def templates_for(self, bay: BayConfig) -> dict[str, tuple[str, SlotTemplate]]:
        """Layout value -> (template name, template) for one bay."""
        return {
            layout: (name, self.slot_templates[name]) for layout, name in bay.slot_templates.items()
        }


def load_config(path: str | Path) -> AppConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppConfig.model_validate(raw)
