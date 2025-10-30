from pydantic import BaseModel
from typing import List, Optional
import numpy as np

BIN_EDGES = np.array([300, 400, 500, 600, 700, 800, 1200], dtype=float)
AM15G_BIN_FRACTIONS = np.array([0.08, 0.19, 0.22, 0.20, 0.14, 0.17], dtype=float)
AM15G_BIN_FRACTIONS = AM15G_BIN_FRACTIONS / AM15G_BIN_FRACTIONS.sum()

CLASS_BANDS = {
    "A+": (0.875, 1.125),
    "A":  (0.75, 1.25),
    "B":  (0.60, 1.40),
    "C":  (0.40, 2.00),
}

class Spectrum(BaseModel):
    wavelength_nm: List[float]
    E: List[float]

class UniformityPayload(BaseModel):
    grid: List[List[float]]

class StabilityPayload(BaseModel):
    irradiance: List[float]
    sample_hz: Optional[float] = 100.0
    sti_window_s: Optional[float] = 1.0
    lti_window_s: Optional[float] = 60.0

def bin_fractions(wavelength, E):
    wl = np.array(wavelength, dtype=float)
    sp = np.array(E, dtype=float)
    idx = np.argsort(wl)
    wl, sp = wl[idx], sp[idx]
    mask = (wl >= BIN_EDGES[0]) & (wl <= BIN_EDGES[-1])
    wl, sp = wl[mask], sp[mask]
    if len(wl) < 2:
        return None
    fracs = []
    for i in range(len(BIN_EDGES)-1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i+1]
        sub_wl = [lo]
        for x in wl[(wl>lo) & (wl<hi)]:
            sub_wl.append(float(x))
        sub_wl.append(hi)
        sub_wl = np.array(sub_wl, dtype=float)
        sub_sp = np.interp(sub_wl, wl, sp)
        val = np.sum((sub_wl[1:]-sub_wl[:-1]) * (sub_sp[:-1]+sub_sp[1:]) * 0.5)
        fracs.append(val)
    fracs = np.array(fracs, dtype=float)
    s = fracs.sum()
    if s > 0:
        fracs = fracs / s
    return fracs

def spectral_class_from_ratios(r):
    for cls in ["A+", "A", "B", "C"]:
        lo, hi = CLASS_BANDS[cls]
        if np.all((r >= lo) & (r <= hi)):
            return cls
    return "Fail"

def uniformity_S(grid):
    g = np.array(grid, dtype=float)
    mx = np.max(g); mn = np.min(g)
    if mx + mn == 0:
        return 1.0
    return (mx - mn) / (mx + mn)

def stability_metrics(series, sample_hz=100.0, sti_window_s=1.0, lti_window_s=60.0):
    x = np.array(series, dtype=float)
    mean = float(np.mean(x)) if len(x) else 0.0
    if mean <= 0: 
        return {"mean": mean, "sti_pct": 100.0, "lti_pct": 100.0}
    def rolling_pct(win_s):
        win = max(1, int(win_s*sample_hz))
        if win <= 1 or win >= len(x):
            return float(np.std(x)/mean*100.0)
        s = np.array([np.std(x[i:i+win]) for i in range(len(x)-win+1)])
        m = np.array([np.mean(x[i:i+win]) for i in range(len(x)-win+1)])
        pct = s/np.maximum(m,1e-9)*100.0
        return float(np.percentile(pct, 95))
    sti = rolling_pct(sti_window_s)
    lti = rolling_pct(lti_window_s)
    return {"mean": mean, "sti_pct": float(sti), "lti_pct": float(lti)}
