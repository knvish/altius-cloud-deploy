
import os
import requests
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="SunSim Suite — Pro (IEC)", page_icon="🌞", layout="wide")

API_URL = os.getenv("API_URL", "http://api:8000")
PUBLIC_API_BASE = os.getenv("PUBLIC_API_BASE", "")

def api(path, method="GET", json=None):
    url = f"{API_URL}{path}"
    r = requests.request(method, url, json=json, timeout=15)
    if not (200 <= r.status_code < 300):
        try:
            err = r.json()
        except Exception:
            err = {"error": r.text}
        st.error(f"API error {r.status_code}: {err}")
        st.stop()
    return r.json()

st.title("🌞 SunSim Suite — Pro (IEC)")
st.caption("Demo • Calibration • IEC Spectral/Uniformity/Stability")

# Tabs: Demo first
tab_demo, tab_sim, tab_hist, tab_health, tab_iec = st.tabs(
    ["Demo: AM1.5G Overlay", "Simulator", "History", "Health", "IEC Test"]
)

# ---------------- DEMO TAB ----------------
with tab_demo:
    st.header("Demo: AM1.5G Reference vs Measured (Auto-loaded)")
    st.caption("Loads from /opt/sunsim-suite-pro-full-iec/demo_data/ if present; otherwise uses built-in arrays.")

    # Built-in defaults
    demo_ref = {
        "wavelength_nm": np.array([300,350,400,450,500,550,600,650,700,750,800,900,1000,1100,1200], dtype=float),
        "AM1.5G":         np.array([0.15,0.45,0.85,1.00,0.95,0.90,0.80,0.65,0.55,0.40,0.30,0.20,0.10,0.05,0.02], dtype=float),
    }
    demo_meas = {
        "wavelength_nm": np.array([300,350,400,450,500,550,600,650,700,750,800,900,1000,1100,1200], dtype=float),
        "E":             np.array([0.05,0.20,0.65,0.85,1.00,0.95,0.80,0.60,0.40,0.30,0.20,0.10,0.05,0.02,0.01], dtype=float),
    }

    # Try CSVs from /opt/.../demo_data
    try:
        ref_csv = "/opt/sunsim-suite-pro-full-iec/demo_data/am15g_demo.csv"
        meas_csv = "/opt/sunsim-suite-pro-full-iec/demo_data/measured_demo.csv"
        if os.path.exists(ref_csv):
            rdf = pd.read_csv(ref_csv)
            if {"wavelength_nm","AM1.5G"}.issubset(rdf.columns):
                demo_ref["wavelength_nm"] = rdf["wavelength_nm"].astype(float).to_numpy()
                demo_ref["AM1.5G"] = rdf["AM1.5G"].astype(float).to_numpy()
        if os.path.exists(meas_csv):
            mdf = pd.read_csv(meas_csv)
            if {"wavelength_nm","E"}.issubset(mdf.columns):
                demo_meas["wavelength_nm"] = mdf["wavelength_nm"].astype(float).to_numpy()
                demo_meas["E"] = mdf["E"].astype(float).to_numpy()
    except Exception:
        pass  # Stick to built-ins on any error

    # Controls with unique keys
    colA, colB, colC = st.columns(3)
    ref_range = colA.selectbox("Plot range (nm)", ["300–1200", "300–4000"], index=0, key="demo_ref_range")
    norm = colB.checkbox("Normalize spectra", value=True, key="demo_norm")
    grid_on = colC.checkbox("Show grid", value=True, key="demo_grid")

    lo, hi = (300,1200) if ref_range == "300–1200" else (300,4000)

    # Crop
    x_ref = demo_ref["wavelength_nm"]; y_ref = demo_ref["AM1.5G"]
    m_ref = (x_ref >= lo) & (x_ref <= hi); x_ref, y_ref = x_ref[m_ref], y_ref[m_ref]
    x_meas = demo_meas["wavelength_nm"]; y_meas = demo_meas["E"]
    m_meas = (x_meas >= lo) & (x_meas <= hi); x_meas, y_meas = x_meas[m_meas], y_meas[m_meas]

    if norm:
        maxv = max(float(np.max(y_ref)) if y_ref.size else 1.0,
                   float(np.max(y_meas)) if y_meas.size else 1.0)
        if maxv > 0:
            y_ref = y_ref / maxv
            y_meas = y_meas / maxv

    fig = plt.figure()
    plt.plot(x_ref, y_ref, label="AM1.5G (Demo)", linewidth=2.2)
    plt.plot(x_meas, y_meas, label="Measured (Demo)", linewidth=1.8, color="red", alpha=0.9)
    plt.xlim(lo, hi)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Irradiance" if norm else "Spectral Irradiance (W m⁻² nm⁻¹)")
    plt.title("Demo: AM1.5G vs Measured")
    plt.legend(frameon=True, framealpha=0.9, loc="upper right")
    if grid_on:
        plt.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.info("For real spectra, use IEC Test → AM1.5G Reference Spectrum Comparison.")

# ---------------- SIMULATOR ----------------
with tab_sim:
    st.subheader("LED currents (mA)")
    col1, col2, col3 = st.columns(3)
    with col1:
        r = st.slider("Red", 0, 1000, 170, 10, key="slider_red")
    with col2:
        g = st.slider("Green", 0, 1000, 400, 10, key="slider_green")
    with col3:
        b = st.slider("Blue", 0, 1000, 400, 10, key="slider_blue")

    if st.button("Simulate", use_container_width=True, key="btn_simulate"):
        data = api("/simulate", method="POST", json={"red": r, "green": g, "blue": b})
        st.success(f"Irradiance: {data['irradiance_wm2']:.1f} W/m²  · run_id={data['run_id']}")

    with st.expander("Calibration", expanded=False):
        cal = api("/calibrate")
        st.write("Active coefficients (W/m² per mA):", cal)
        rc = st.number_input("red_coeff", value=float(cal["red_coeff"]), step=0.01, format="%.4f", key="num_red_coeff")
        gc = st.number_input("green_coeff", value=float(cal["green_coeff"]), step=0.01, format="%.4f", key="num_green_coeff")
        bc = st.number_input("blue_coeff", value=float(cal["blue_coeff"]), step=0.01, format="%.4f", key="num_blue_coeff")
        bias = st.number_input("bias", value=float(cal["bias"]), step=0.1, format="%.2f", key="num_bias")
        note = st.text_input("note", value="manual update", key="txt_note")
        if st.button("Update calibration", key="btn_update_cal"):
            api("/calibrate", method="POST", json={"red_coeff": rc, "green_coeff": gc, "blue_coeff": bc, "bias": bias, "note": note})
            st.success("Calibration updated.")

# ---------------- HISTORY ----------------
with tab_hist:
    st.subheader("Recent runs")
    items = api("/runs?limit=200")["items"]
    if items:
        df = pd.DataFrame(items)
        st.dataframe(df, use_container_width=True)
        st.caption("Persisted in Postgres; exported via /runs.")
    else:
        st.info("No runs yet. Simulate above.")

# ---------------- HEALTH ----------------
with tab_health:
    st.subheader("Health")
    h = api("/health")
    st.write(h)
    if PUBLIC_API_BASE:
        st.caption(f"External API base (LAN/Tailnet): {PUBLIC_API_BASE}")

# ---------------- IEC TESTS ----------------
with tab_iec:
    st.header("IEC Test — Spectral, Uniformity, Stability")
    st.write(
        """This module helps you visualize **IEC 60904-9** metrics:
- **Spectral Match** vs AM1.5G (6 bins: 300–1200 nm) with ratio bands for A+/A/B/C
- **Spatial Non-uniformity** heatmap and S (%)
- **Temporal Instability** time series with STI/LTI (%)."""
    )

    # --- Spectral (bins) ---
    st.subheader("1) Spectral Match (CSV: wavelength_nm,E)")
    spec_file = st.file_uploader("Upload spectrum CSV", type=["csv"], key="spec_csv_iec")
    if spec_file is not None:
        df = pd.read_csv(spec_file)
        if {"wavelength_nm","E"}.issubset(df.columns):
            wl = df["wavelength_nm"].to_numpy()
            E = df["E"].to_numpy()
            res = api("/iec/spectral", method="POST", json={"wavelength_nm": wl.tolist(), "E": E.tolist()})
            if res.get("ok"):
                fr = np.array(res["bin_fractions"])
                am = np.array(res["am15g_bin_fractions"])
                ratio = np.array(res["ratio"])
                edges = np.array(res["bin_edges_nm"])

                st.success(f"Spectral Class (bins): {res['class']}")

                fig1 = plt.figure()
                E_norm = E / np.maximum(np.max(E), 1e-9)
                plt.plot(wl, E_norm, label="Measured (normalized)")

                am_wl, am_y = [], []
                for i in range(len(edges) - 1):
                    lo, hi = edges[i], edges[i+1]
                    width = max(1e-9, hi - lo)
                    density = am[i] / width
                    am_wl.extend([lo, hi])
                    am_y.extend([density, density])
                am_y = np.array(am_y, dtype=float)
                am_y_norm = am_y / np.maximum(am_y.max(), 1e-9)
                plt.plot(am_wl, am_y_norm, linestyle="--", label="AM1.5G (bin-step, normalized)")
                for x in edges:
                    plt.axvline(x=x, linestyle=":", linewidth=0.8)

                plt.title("Measured Spectrum vs AM1.5G (normalized)")
                plt.xlabel("Wavelength (nm)")
                plt.ylabel("Relative")
                plt.legend()
                st.pyplot(fig1)

            else:
                st.error(res.get("error","Unable to compute spectral match."))
        else:
            st.error("CSV must have columns: wavelength_nm,E")

    # --- Uniformity ---
    st.subheader("2) Spatial Uniformity (CSV grid, no header)")
    grid_file = st.file_uploader("Upload uniformity grid", type=["csv"], key="grid_csv_iec")
    if grid_file is not None:
        gdf = pd.read_csv(grid_file, header=None)
        grid = gdf.values.tolist()
        res = api("/iec/uniformity", method="POST", json={"grid": grid})
        if res.get("ok"):
            S = res["S"]; cls = res["class"]
            st.success(f"Non-uniformity S = {S*100:.2f}% → Class {cls}")
            fig = plt.figure()
            plt.imshow(gdf.values, aspect="equal")
            plt.colorbar(label="Irradiance")
            plt.title("Uniformity Heatmap")
            st.pyplot(fig)
        else:
            st.error("Could not compute uniformity.")

    # --- Stability ---
    st.subheader("3) Temporal Instability (single-column CSV)")
    colA,colB,colC = st.columns(3)
    sample_hz = colA.number_input("Sample rate (Hz)", 1.0, 5000.0, 100.0, 1.0, key="num_sample_hz")
    sti_s = colB.number_input("STI window (s)", 0.1, 10.0, 1.0, 0.1, key="num_sti_window")
    lti_s = colC.number_input("LTI window (s)", 5.0, 600.0, 60.0, 5.0, key="num_lti_window")
    ts_file = st.file_uploader("Upload time-series CSV", type=["csv"], key="ts_csv_iec")
    if ts_file is not None:
        ts = pd.read_csv(ts_file, header=None)[0].astype(float).to_numpy()
        res = api("/iec/stability", method="POST", json={
            "irradiance": ts.tolist(), "sample_hz": sample_hz, "sti_window_s": sti_s, "lti_window_s": lti_s
        })
        if res.get("ok", True):
            st.success(f"STI ≈ {res['sti_pct']:.2f}% (class {res['class_sti']}), LTI ≈ {res['lti_pct']:.2f}% (class {res['class_lti']})")
            fig = plt.figure()
            plt.plot(ts)
            plt.title("Irradiance vs Time"); plt.xlabel("Sample"); plt.ylabel("Irradiance")
            st.pyplot(fig)
        else:
            st.error("Could not compute stability metrics.")

    # --- AM1.5G full-spectrum comparison (manual upload, real data) ---
    st.subheader("4) AM1.5G Reference Spectrum Comparison")
    st.caption("Upload ASTM G-173 CSV (must include columns: wavelength_nm, AM1.5G); optionally overlay your measured spectrum.")

    ref_csv_up = st.file_uploader("Upload ASTM G-173 reference CSV", type=["csv"], key="ref_am15g_up")
    meas_csv_up = st.file_uploader("Upload your measured spectrum CSV (wavelength_nm,E)", type=["csv"], key="meas_am15g_up")

    ref_range2 = st.selectbox("Plot range (nm)", ["300–1200", "300–4000"], index=0, key="ref_range_iec")
    norm2 = st.checkbox("Normalize spectra", value=True, key="norm_iec")

    if ref_csv_up is not None:
        refdf = pd.read_csv(ref_csv_up)
        if {"wavelength_nm", "AM1.5G"}.issubset(refdf.columns):
            x = refdf["wavelength_nm"].astype(float).to_numpy()
            y = refdf["AM1.5G"].astype(float).to_numpy()
            lo2, hi2 = (300, 1200) if ref_range2 == "300–1200" else (300, 4000)
            m2 = (x >= lo2) & (x <= hi2)
            x, y = x[m2], y[m2]
            if norm2:
                y = y / np.maximum(np.max(y), 1e-9)
            fig = plt.figure()
            plt.plot(x, y, label="AM1.5G (ASTM G-173)", linewidth=2.2)
            if meas_csv_up is not None:
                mdf = pd.read_csv(meas_csv_up)
                if {"wavelength_nm", "E"}.issubset(mdf.columns):
                    xm = mdf["wavelength_nm"].astype(float).to_numpy()
                    ym = mdf["E"].astype(float).to_numpy()
                    mm = (xm >= lo2) & (xm <= hi2)
                    xm, ym = xm[mm], ym[mm]
                    if norm2:
                        ym = ym / np.maximum(np.max(ym), 1e-9)
                    plt.plot(xm, ym, label="Measured", linewidth=1.8, color="red", alpha=0.9)
                else:
                    st.warning("Measured CSV must have columns: wavelength_nm,E")
            plt.xlim(lo2, hi2)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Spectral Irradiance (W m⁻² nm⁻¹)" if not norm2 else "Normalized Irradiance")
            plt.title("AM1.5G Reference vs Measured Spectrum")
            plt.legend(frameon=True, framealpha=0.9, loc="upper right")
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.error("CSV must contain columns: wavelength_nm and AM1.5G")
    else:
        st.info("Upload the ASTM G-173 file to plot the AM1.5G reference curve.")
