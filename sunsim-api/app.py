import os, uuid
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

# IEC imports
from iec import (
    Spectrum, UniformityPayload, StabilityPayload,
    bin_fractions, AM15G_BIN_FRACTIONS, spectral_class_from_ratios,
    uniformity_S, stability_metrics, BIN_EDGES
)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://sunsim:sunsim_pw@postgres:5432/sunsimdb")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="SunSim API", version="0.2.0")

class RGB(BaseModel):
    red: int = Field(ge=0, le=1000)
    green: int = Field(ge=0, le=1000)
    blue: int = Field(ge=0, le=1000)

class Calib(BaseModel):
    red_coeff: float
    green_coeff: float
    blue_coeff: float
    bias: float = 0.0
    note: Optional[str] = None

def get_active_calibration():
    with engine.begin() as conn:
        row = conn.execute(text("SELECT red_coeff, green_coeff, blue_coeff, bias FROM calibration ORDER BY id DESC LIMIT 1")).first()
        if row is None:
            return (0.18, 0.22, 0.20, 0.0)
        return tuple(row)

@app.get("/health")
def health():
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"ok": True, "db": db_ok}

@app.post("/calibrate")
def calibrate(c: Calib):
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO calibration (red_coeff, green_coeff, blue_coeff, bias, note) VALUES (:r,:g,:b,:bias,:note)"),
            {"r": c.red_coeff, "g": c.green_coeff, "b": c.blue_coeff, "bias": c.bias, "note": c.note}
        )
    return {"ok": True}

@app.get("/calibrate")
def get_calibration():
    rc, gc, bc, bias = get_active_calibration()
    return {"red_coeff": rc, "green_coeff": gc, "blue_coeff": bc, "bias": bias}

@app.post("/simulate")
def simulate(rgb: RGB):
    rc, gc, bc, bias = get_active_calibration()
    irr = rc*rgb.red + gc*rgb.green + bc*rgb.blue + bias
    irr = float(max(0.0, irr))
    run_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO runs (id, red_ma, green_ma, blue_ma, irradiance_wm2, model_version) VALUES (:id, :r, :g, :b, :irr, 'v0-starter')"
        ), {"id": run_id, "r": rgb.red, "g": rgb.green, "b": rgb.blue, "irr": irr})
    return {"irradiance_wm2": irr, "run_id": run_id}

@app.get("/runs")
def list_runs(limit: int = 50):
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT id, ts, red_ma, green_ma, blue_ma, irradiance_wm2 FROM runs ORDER BY ts DESC LIMIT :lim"
        ), {"lim": limit}).mappings().all()
    return {"items": [dict(r) for r in rows]}

# ------------- IEC endpoints -------------
@app.post("/iec/spectral")
def iec_spectral(spec: Spectrum):
    fr = bin_fractions(spec.wavelength_nm, spec.E)
    if fr is None:
        return {"ok": False, "error": "Spectrum outside 300–1200 nm or too few points"}
    ratio = fr / AM15G_BIN_FRACTIONS
    cls = spectral_class_from_ratios(ratio)
    return {"ok": True, "bin_fractions": fr.tolist(), "am15g_bin_fractions": AM15G_BIN_FRACTIONS.tolist(), "ratio": ratio.tolist(), "class": cls, "bin_edges_nm": BIN_EDGES.tolist()}

@app.post("/iec/uniformity")
def iec_uniformity(payload: UniformityPayload):
    S = uniformity_S(payload.grid)
    cls = "A+" if S <= 0.01 else ("A" if S <= 0.02 else ("B" if S <= 0.05 else ("C" if S <= 0.10 else "Fail")))
    return {"ok": True, "S": float(S), "class": cls}

@app.post("/iec/stability")
def iec_stability(payload: StabilityPayload):
    m = stability_metrics(payload.irradiance, payload.sample_hz or 100.0, payload.sti_window_s or 1.0, payload.lti_window_s or 60.0)
    sti, lti = m["sti_pct"], m["lti_pct"]
    def grade(v, a_plus, a):
        return "A+" if v <= a_plus else ("A" if v <= a else ("B" if v <= a*2.5 else "C"))
    cls_sti = grade(sti, 0.25, 0.5)
    cls_lti = grade(lti, 1.0, 2.0)
    return {"ok": True, **m, "class_sti": cls_sti, "class_lti": cls_lti}
