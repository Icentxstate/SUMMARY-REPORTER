# -*- coding: utf-8 -*-
# WSR Graph Generator – JMP-style figures
# Run: streamlit run wsr_streamlit_app.py

import os
from io import BytesIO
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --------------------
# CONFIG (edit if needed)
# --------------------
WQS = {
    "DO_min": 5.0,            # mg/L
    "pH_min": 6.5,            # s.u.
    "pH_max": 9.0,            # s.u.
    "TDS_max": 500.0,         # mg/L
    "WT_max": 32.2            # °C
}

FIG_DIR = "wsr_figures"
os.makedirs(FIG_DIR, exist_ok=True)

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="WSR Graph Generator (JMP Style)", layout="wide")
st.title("📊 Watershed Summary Report – JMP-style Graphs")

uploaded = st.file_uploader("Upload the dataset (.xlsx)", type=["xlsx"])

# --------------------
# Helpers
# --------------------
def coerce_num(s):
    return pd.to_numeric(s, errors="coerce")

def pick(df, *names, default=None):
    for n in names:
        if n in df.columns:
            return n
    return default

def prepare_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # ---- Standardize column names we need
    date_col = pick(df, "Sample Date", "Date", "SampleDate")
    if date_col is None:
        raise ValueError("Missing 'Sample Date' column.")
    df["Sample Date"] = pd.to_datetime(df[date_col], errors="coerce")

    site_col = pick(df, "Site ID", "Site", "SiteID", "Site ID: Site Name")
    if site_col is None:
        raise ValueError("Missing 'Site ID' column.")
    df["Site ID"] = df[site_col].astype(str)

    # Core params (use explicit names from your JMP export; fallbacks included)
    df["DO"]  = coerce_num(df.get("Dissolved Oxygen (mg/L) Average"))
    df["pH"]  = coerce_num(df.get("pH (standard units)"))
    df["TDS"] = coerce_num(df.get("TDS (mg/L)"))

    # If TDS missing but Conductivity exists → TDS≈0.65*Cond
    if df["TDS"].isna().all():
        cond_col = pick(df, "Conductivity (µS/cm)", "Conductivity (?S/cm)")
        if cond_col:
            df["TDS"] = 0.65 * coerce_num(df[cond_col])

    df["Secchi"] = coerce_num(df.get("Secchi Disk Transparency - Average"))
    df["Tube"]   = coerce_num(df.get("Transparency Tube (meters)"))

    wt_col = pick(df, "Water Temperature (° C)", "Water Temp Rounded")
    df["WT"] = coerce_num(df.get(wt_col)) if wt_col else np.nan

    depth_col = pick(df, "Total Depth (meters)", "Total Depth (m)")
    df["Depth"] = coerce_num(df.get(depth_col)) if depth_col else np.nan

    # Keep only used columns
    df = df[["Sample Date","Site ID","DO","pH","TDS","Secchi","Tube","WT","Depth"]]
    # Sort Site IDs numerically when possible (to match figures)
    try:
        order = sorted(df["Site ID"].unique(), key=lambda x: int(str(x)))
    except:
        order = sorted(df["Site ID"].unique())
    df["Site ID"] = pd.Categorical(df["Site ID"], categories=order, ordered=True)
    return df

def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path

def boxplot_by_site(ax, data, y, ylabel, wqs_value=None, wqs_label="WQS",
                    wqs_min=None, wqs_max=None, ylim=None):
    # Prepare groups
    groups = [data[data["Site ID"]==cat][y].dropna().values for cat in data["Site ID"].cat.categories]
    positions = np.arange(1, len(groups)+1)

    # Boxplot – white fill, black edges, small fliers
    bp = ax.boxplot(groups, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops=dict(marker='o', markersize=4, markerfacecolor='black'))
    for patch in bp['boxes']:
        patch.set(facecolor='white', edgecolor='black', linewidth=1.2)
    for whisk in bp['whiskers'] + bp['caps'] + bp['medians']:
        whisk.set(color='black', linewidth=1.2)

    ax.set_xticks(positions)
    ax.set_xticklabels(list(data["Site ID"].cat.categories))
    ax.set_ylabel(ylabel)

    # WQS lines
    if wqs_value is not None:
        ax.axhline(wqs_value, ls='--', color='red', linewidth=1.3)
        ax.text(0.35, wqs_value + (ylim[1]-ylim[0])*0.005 if ylim else wqs_value*1.005,
                wqs_label, color='red')

    if wqs_min is not None:
        ax.axhline(wqs_min, ls='--', color='red', linewidth=1.3)
        ax.text(0.35, wqs_min + (ylim[1]-ylim[0])*0.005 if ylim else wqs_min*1.005,
                "WQS Min", color='red')
    if wqs_max is not None:
        ax.axhline(wqs_max, ls='--', color='red', linewidth=1.3)
        ax.text(0.35, wqs_max + (ylim[1]-ylim[0])*0.005 if ylim else wqs_max*1.005,
                "WQS Max", color='red')

    if ylim: ax.set_ylim(*ylim)
    ax.grid(False)

def transparency_dual_boxplot(ax, df):
    cats = list(df["Site ID"].cat.categories)
    pos = np.arange(1, len(cats)+1)
    off = 0.18

    secchi_groups = [df[df["Site ID"]==c]["Secchi"].dropna().values for c in cats]
    tube_groups   = [df[df["Site ID"]==c]["Tube"].dropna().values   for c in cats]

    # Secchi (blue outline)
    bp1 = ax.boxplot(secchi_groups, positions=pos-off, widths=0.32, patch_artist=True,
                     showfliers=True, flierprops=dict(marker='o', markersize=4,
                                                      markerfacecolor='white', markeredgecolor='blue'))
    for b in bp1['boxes']:
        b.set(facecolor='white', edgecolor='blue', linewidth=1.4)
    for k in ['whiskers','caps','medians']:
        for l in bp1[k]: l.set(color='blue', linewidth=1.2)

    # Tube (red outline)
    bp2 = ax.boxplot(tube_groups, positions=pos+off, widths=0.32, patch_artist=True,
                     showfliers=True, flierprops=dict(marker='o', markersize=4,
                                                      markerfacecolor='white', markeredgecolor='red'))
    for b in bp2['boxes']:
        b.set(facecolor='white', edgecolor='red', linewidth=1.4)
    for k in ['whiskers','caps','medians']:
        for l in bp2[k]: l.set(color='red', linewidth=1.2)

    ax.set_xticks(pos)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Transparency (meters)")
    ax.set_ylim(0, 0.62)
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ["Secchi Disk", "Transparency Tube"],
              loc='upper right', frameon=False)

def water_temp_scatter(ax, df):
    # one symbol per site; keep legend simple
    for s in df["Site ID"].cat.categories:
        sub = df[df["Site ID"]==s]
        ax.scatter(sub["Sample Date"], sub["WT"], s=28, label=str(s))
    ax.axhline(WQS["WT_max"], ls='--', color='red', linewidth=1.3)
    ax.text(df["Sample Date"].min(), WQS["WT_max"]+0.4, "WQS", color='red')
    ax.set_xlabel("Sample Date")
    ax.set_ylabel("Water Temperature (°C)")
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(title="Site ID", bbox_to_anchor=(1.02, 0.5), loc="center left")

def table6(df):
    # sites with >=10 events
    counts = df.groupby("Site ID").size()
    keep_sites = counts[counts >= 10].index
    sub = df[df["Site ID"].isin(keep_sites)]

    params = [
        ("Air Temperature (°C)", None), # placeholder to keep format similar (may be NA)
        ("Water Temperature (°C)", "WT"),
        ("Dissolved Oxygen (mg/L)", "DO"),
        ("pH (standard units)", "pH"),
        ("TDS (mg/L)", "TDS"),
        ("Secchi Disk Transparency (m)", "Secchi"),
        ("Transparency Tube (m)", "Tube"),
        ("Total Depth (m)", "Depth"),
    ]
    rows = []
    for label, col in params:
        for stat in ["Mean","Std Dev","Range"]:
            row = {"Parameter": label if stat=="Mean" else "", "Statistic": stat}
            for s in keep_sites:
                if col is None:
                    val = "ND"
                else:
                    vals = sub.loc[sub["Site ID"]==s, col].dropna()
                    if len(vals) < 10:
                        val = "ND"
                    else:
                        if stat=="Mean":    val = round(float(vals.mean()), 2)
                        if stat=="Std Dev": val = round(float(vals.std(ddof=1)), 2)
                        if stat=="Range":   val = round(float(vals.max()-vals.min()), 2)
                row[str(s)] = val
            rows.append(row)
    return pd.DataFrame(rows)

# --------------------
# Main
# --------------------
if uploaded:
    raw = pd.read_excel(uploaded)
    df = prepare_dataframe(raw)

    # ---- Figure: DO boxplot
    fig1, ax1 = plt.subplots(figsize=(8.5, 5.2))
    boxplot_by_site(ax1, df, "DO", "Dissolved Oxygen (mg/L)",
                    wqs_value=WQS["DO_min"], wqs_label="WQS",
                    ylim=(4.5, 14.2))
    savefig(fig1, "Figure_DO_Boxplot.png")

    # ---- Figure: pH boxplot (Min/Max)
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.2))
    boxplot_by_site(ax2, df, "pH", "pH (standard units)",
                    wqs_min=WQS["pH_min"], wqs_max=WQS["pH_max"],
                    ylim=(6.4, 9.2))
    savefig(fig2, "Figure_pH_Boxplot.png")

    # ---- Figure: TDS boxplot
    fig3, ax3 = plt.subplots(figsize=(8.5, 5.2))
    boxplot_by_site(ax3, df, "TDS", "Total Dissolved Solids (mg/L)",
                    wqs_value=WQS["TDS_max"], wqs_label="WQS",
                    ylim=(0, max(900, np.nanmax(df["TDS"])*1.1)))
    savefig(fig3, "Figure_TDS_Boxplot.png")

    # ---- Figure: Transparency dual boxplot
    fig4, ax4 = plt.subplots(figsize=(11.5, 5.2))
    transparency_dual_boxplot(ax4, df)
    savefig(fig4, "Figure_Transparency_DualBoxplot.png")

    # ---- Figure: Water Temp scatter over time
    fig5, ax5 = plt.subplots(figsize=(10.5, 6))
    water_temp_scatter(ax5, df)
    savefig(fig5, "Figure_WaterTemp_Time.png")

    # ---- Table 6
    t6 = table6(df)
    csv_path = os.path.join(FIG_DIR, "Table6_Summary.csv")
    t6.to_csv(csv_path, index=False)

    # ---------------- Show in Streamlit ----------------
    st.subheader("Figure. Dissolved Oxygen by Site")
    st.pyplot(fig1)

    st.subheader("Figure. pH by Site")
    st.pyplot(fig2)

    st.subheader("Figure. Total Dissolved Solids by Site")
    st.pyplot(fig3)

    st.subheader("Figure. Transparency by Site (Secchi vs Tube)")
    st.pyplot(fig4)

    st.subheader("Figure. Water Temperature Over Time")
    st.pyplot(fig5)

    st.markdown("## 📋 Table 6 – Summary Statistics")
    st.dataframe(t6, use_container_width=True)

    # -------------- ZIP download --------------
    st.markdown("## 📦 Download All Outputs")
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in os.listdir(FIG_DIR):
            z.write(os.path.join(FIG_DIR, f), arcname=f)
    buf.seek(0)
    st.download_button(
        "📥 Download ZIP (figures + Table 6)",
        data=buf,
        file_name="WSR_All_Results.zip",
        mime="application/zip"
    )
    st.success("✅ Figures and Table 6 generated.")
else:
    st.info("Upload your JMP-formatted Excel to generate the figures.")
