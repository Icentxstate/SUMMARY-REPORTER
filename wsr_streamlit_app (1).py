# WSR Graph Generator (Exact Style, Excel-driven Climate, unified marker style)
# -------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, zipfile
from io import BytesIO

# ================== Streamlit Page ==================
st.set_page_config(page_title='WSR Graph Generator', layout='wide')
st.title("📊 Watershed Summary Report Graph Generator (Exact Style)")

uploaded_file = st.file_uploader("Upload your Excel dataset (.xlsx)", type="xlsx")

# ================== Helpers ==================
def get_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([np.nan]*len(df), index=df.index)

def to_num(s):
    return pd.to_numeric(s, errors='coerce')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')

def style_axes(ax, xlabel='', ylabel='', site_order=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_color('black')
    if site_order is not None:
        ax.set_xticks(range(1, len(site_order)+1))
        ax.set_xticklabels(site_order)

def series_by_site(df, site_order, ycol):
    return [df.loc[df['Site ID'].eq(s), ycol].dropna().values for s in site_order]

def find_first_name(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

# ===== Monthly Climate builder (from Excel) =====
def build_monthly_climate_from_df(df):
    date_col = find_first_name(df, ['Sample Date', 'Date', 'SampleDate', 'Datetime'])
    if date_col is None:
        return None

    temp_name = find_first_name(df, [
        'Air Temperature (° C)', 'Air Temperature (°C)',
        'Water Temperature (° C)', 'Water Temperature (°C)'
    ])
    ppt_name  = find_first_name(df, [
        'Rainfall Accumulation', 'Precipitation', 'Rain', 'Rain (in)'
    ])

    tmp = df[[date_col]].copy()
    tmp['__date__'] = pd.to_datetime(tmp[date_col], errors='coerce')
    tmp = tmp.dropna(subset=['__date__']).sort_values('__date__')

    if temp_name:
        tmp['__temp__'] = to_num(df[temp_name])
    if ppt_name:
        tmp['__ppt__'] = to_num(df[ppt_name])

    if ('__temp__' not in tmp and '__ppt__' not in tmp) or \
       (tmp.get('__temp__', pd.Series(dtype=float)).notna().sum() == 0 and
        tmp.get('__ppt__',  pd.Series(dtype=float)).notna().sum()  == 0):
        return None

    monthly = pd.DataFrame({'MonthNum': range(1, 13)})

    if '__temp__' in tmp and tmp['__temp__'].notna().any():
        t_month = (tmp.dropna(subset=['__temp__'])
                     .set_index('__date__')['__temp__']
                     .resample('M').mean())
        t_month = (t_month.reset_index()
                           .assign(MonthNum=lambda d: d['__date__'].dt.month)
                           .groupby('MonthNum', as_index=False)['__temp__'].mean()
                           .rename(columns={'__temp__': 'TempMeanC'}))
        monthly = monthly.merge(t_month, on='MonthNum', how='left')

    if '__ppt__' in tmp and tmp['__ppt__'].notna().any():
        p_month = (tmp.dropna(subset=['__ppt__'])
                     .set_index('__date__')['__ppt__']
                     .resample('M').sum())
        p_month = (p_month.reset_index()
                           .assign(MonthNum=lambda d: d['__date__'].dt.month)
                           .groupby('MonthNum', as_index=False)['__ppt__'].sum()
                           .rename(columns={'__ppt__': 'Precip'}))
        monthly = monthly.merge(p_month, on='MonthNum', how='left')

    return monthly.sort_values('MonthNum')

# ================== Main ==================
if uploaded_file:
    # ---------- Read ----------
    df = pd.read_excel(uploaded_file)

    # ---------- Prepare columns ----------
    df['Sample Date'] = pd.to_datetime(get_col(df, 'Sample Date', 'Date', 'SampleDate'), errors='coerce')
    df['Site ID']     = get_col(df, 'Site ID: Site Name', 'Site ID', 'Station ID').astype(str)

    df['Air Temp Rounded']   = to_num(get_col(df, 'Air Temperature (° C)', 'Air Temperature (°C)', 'Air Temp Rounded'))
    df['Water Temp Rounded'] = to_num(get_col(df, 'Water Temp Rounded', 'Water Temperature (° C)', 'Water Temperature (°C)'))

    cond_col = get_col(df, 'Conductivity (µS/cm)', 'Conductivity (uS/cm)', 'Conductivity (?S/cm)', 'Conductivity')
    df['Conductivity'] = to_num(cond_col)

    tds_existing = get_col(df, 'TDS (mg/L)', 'Total Dissolved Solids (mg/L)')
    if tds_existing.notna().sum() > 0:
        df['TDS (mg/L)'] = to_num(tds_existing)
    else:
        df['TDS (mg/L)'] = df['Conductivity'] * 0.65

    df['DO_avg'] = to_num(get_col(df, 'Dissolved Oxygen (mg/L) Average', 'DO_avg', 'Dissolved Oxygen (mg/L)'))
    df['pH']     = to_num(get_col(df, 'pH Rounded', 'pH (standard units)', 'pH'))

    df['Secchi']            = to_num(get_col(df, 'Secchi Disk Transparency - Average', 'Secchi Disk Transparency (m)', 'Secchi'))
    df['Transparency Tube'] = to_num(get_col(df, 'Transparency Tube (meters)', 'Transparency Tube (m)'))
    df['Total Depth']       = to_num(get_col(df, 'Total Depth (meters)', 'Total Depth (m)', 'Depth (m)'))

    # Drop rows without Site ID
    df = df[df['Site ID'].notna() & df['Site ID'].ne('')].copy()

    # ---------- Output dir ----------
    output_dir = "wsr_figures"
    ensure_dir(output_dir)

    # ---------- WQS Constants ----------
    WQS_TEMP = 32.2
    WQS_TDS  = 500
    WQS_DO   = 5.0
    WQS_pH_MIN, WQS_pH_MAX = 6.5, 9.0

    # Site order
    site_order = list(pd.unique(df['Site ID']))

    # ================== Figure 6: Water Temperature (scatter, all circles) ==================
    fig6, ax = plt.subplots(figsize=(14, 6))
    for s in site_order:
        dsi = df[df['Site ID'].eq(s)]
        ax.scatter(
            dsi['Sample Date'],
            dsi['Water Temp Rounded'],
            s=40,
            marker='o',   # همه دایره
            label=s,
            alpha=0.9
        )
    ax.axhline(WQS_TEMP, linestyle='--', color='red', linewidth=1.5)
    if df['Sample Date'].notna().any():
        xmin = df['Sample Date'].min()
        ax.text(xmin, WQS_TEMP + 0.5, 'WQS', color='red', va='bottom')
    ax.set_xlabel('Sample Date'); ax.set_ylabel('Water Temperature (°C)')
    ax.legend(title='Site ID', loc='center left', bbox_to_anchor=(1.0, 0.5))
    save_figure(fig6, os.path.join(output_dir, "Figure6_WaterTemperature.png"))

    # ================== Figure 7: TDS ==================
    fig7, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(series_by_site(df, site_order, 'TDS (mg/L)'),
               patch_artist=False, whis=1.5,
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               boxprops=dict(color='black'),
               flierprops=dict(marker='o', markersize=4,
                               markerfacecolor='black', markeredgecolor='black'))
    style_axes(ax, 'Site ID', 'TDS (mg/L)', site_order)
    ax.axhline(WQS_TDS, linestyle='--', color='red')
    ax.text(0.5, WQS_TDS + 10, 'WQS', color='red')
    save_figure(fig7, os.path.join(output_dir, "Figure7_TDS_Boxplot.png"))

    # ================== Figure 8: DO ==================
    fig8, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(series_by_site(df, site_order, 'DO_avg'),
               patch_artist=False, whis=1.5,
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               boxprops=dict(color='black'),
               flierprops=dict(marker='o', markersize=4,
                               markerfacecolor='black', markeredgecolor='black'))
    style_axes(ax, 'Site ID', 'Dissolved Oxygen (mg/L)', site_order)
    ax.axhline(WQS_DO, linestyle='--', color='red')
    ax.text(0.5, WQS_DO + 0.1, 'WQS', color='red')
    save_figure(fig8, os.path.join(output_dir, "Figure8_DO_Boxplot.png"))

    # ================== Figure 9: pH ==================
    fig_ph, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(series_by_site(df, site_order, 'pH'),
               patch_artist=False, whis=1.5,
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               boxprops=dict(color='black'),
               flierprops=dict(marker='o', markersize=4,
                               markerfacecolor='black', markeredgecolor='black'))
    style_axes(ax, 'Site ID', 'pH (standard units)', site_order)
    ax.axhline(WQS_pH_MAX, linestyle='--', color='red')
    ax.axhline(WQS_pH_MIN, linestyle='--', color='red')
    ax.text(0.5, WQS_pH_MAX + 0.03, 'WQS Max', color='red')
    ax.text(0.5, WQS_pH_MIN + 0.03, 'WQS Min', color='red')
    save_figure(fig_ph, os.path.join(output_dir, "Figure9_pH_Boxplot.png"))

    # ================== Figure 10: Transparency ==================
    trans_df = df.melt(id_vars=['Site ID'],
                       value_vars=['Secchi', 'Transparency Tube'],
                       var_name='Type', value_name='Value').dropna()

    fig10, ax = plt.subplots(figsize=(12, 6))
    pos = np.arange(1, len(site_order)+1)
    offset = 0.18
    type2shift = {'Secchi': -offset, 'Transparency Tube': +offset}
    type2color = {'Secchi': 'blue', 'Transparency Tube': 'red'}

    for t in ['Secchi', 'Transparency Tube']:
        data_t = [trans_df[(trans_df['Site ID'].eq(s)) & (trans_df['Type'].eq(t))]['Value']
                  .dropna().values for s in site_order]
        ax.boxplot(
            data_t,
            positions=pos + type2shift[t],
            widths=0.28,
            patch_artist=True, whis=1.5,
            medianprops=dict(color='black'),
            whiskerprops=dict(color=type2color[t]),
            capprops=dict(color=type2color[t]),
            boxprops=dict(facecolor='white', edgecolor=type2color[t]),
            flierprops=dict(marker='o', markersize=4,
                            markerfacecolor=type2color[t], markeredgecolor='black')
        )

    style_axes(ax, 'Site ID', 'Transparency (meters)', site_order)
    ax.set_ylim(0, 0.62)
    handles = [plt.Line2D([0], [0], color='blue', lw=2, label='Secchi Disk'),
               plt.Line2D([0], [0], color='red',  lw=2, label='Transparency Tube')]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5))
    save_figure(fig10, os.path.join(output_dir, "Figure10_Transparency_Boxplot.png"))

    # ================== Figure 11: Total Depth ==================
    fig11, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(series_by_site(df, site_order, 'Total Depth'),
               patch_artist=False, whis=1.5,
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               boxprops=dict(color='black'),
               flierprops=dict(marker='o', markersize=4,
                               markerfacecolor='black', markeredgecolor='black'))
    style_axes(ax, 'Site ID', 'Total Depth (m)', site_order)
    save_figure(fig11, os.path.join(output_dir, "Figure11_TotalDepth_Boxplot.png"))

    # ================== Monthly Climate (from Excel) ==================
    monthly_climate = build_monthly_climate_from_df(df)
    if monthly_climate is not None:
        month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        monthly_climate['Month'] = monthly_climate['MonthNum'].map(dict(enumerate(month_labels, start=1)))

        fig_climate, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Month')
        if 'TempMeanC' in monthly_climate:
            ax1.set_ylabel('Temperature (°C)', color='red')
            ax1.plot(monthly_climate['Month'], monthly_climate['TempMeanC'], color='red', linewidth=3)
            ax1.tick_params(axis='y', labelcolor='red')
        ax2 = ax1.twinx()
        if 'Precip' in monthly_climate:
            ax2.set_ylabel('Precipitation', color='blue')
            ax2.bar(monthly_climate['Month'], monthly_climate['Precip'], alpha=0.7, color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
        fig_climate.tight_layout()
        save_figure(fig_climate, os.path.join(output_dir, "Figure_MonthlyClimate.png"))
    else:
        fig_climate = None
        st.warning("⚠️ اقلیم ماهانه پیدا نشد یا ستون‌های لازم وجود ندارد.")

    # ================== ZIP download ==================
    st.markdown("## 📦 Download All Results (Figures + Table 6)")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for f in os.listdir(output_dir):
            zipf.write(os.path.join(output_dir, f), arcname=f)
    zip_buffer.seek(0)
    st.download_button("📥 Download ZIP", data=zip_buffer,
                       file_name="WSR_All_Results.zip", mime="application/zip")

    # ================== Show charts ==================
    st.subheader("Figure 6. Water Temperature Over Time by Site")
    st.pyplot(fig6); plt.close(fig6)

    st.subheader("Figure 7. TDS by Site")
    st.pyplot(fig7); plt.close(fig7)

    st.subheader("Figure 8. Dissolved Oxygen by Site")
    st.pyplot(fig8); plt.close(fig8)

    st.subheader("Figure 9. pH by Site")
    st.pyplot(fig_ph); plt.close(fig_ph)

    st.subheader("Figure 10. Transparency by Site")
    st.pyplot(fig10); plt.close(fig10)

    st.subheader("Figure 11. Total Depth by Site")
    st.pyplot(fig11); plt.close(fig11)

    if fig_climate is not None:
        st.subheader("Figure #: Monthly Avg Temperature and Total Precipitation")
        st.pyplot(fig_climate); plt.close(fig_climate)
