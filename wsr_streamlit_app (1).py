import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import io
import zipfile
from io import BytesIO

# =============================
# Page config
# =============================
st.set_page_config(page_title='WSR Graph Generator', layout='wide')
st.title('📊 Watershed Summary Report Graph Generator — JMP-Style Match')

# =============================
# Sidebar Controls
# =============================
st.sidebar.header('⚙️ Options')
min_events_per_site = st.sidebar.number_input('Minimum events per site (for Table 6)', min_value=1, value=10, step=1)
make_pdf = st.sidebar.checkbox('Also create a multi-page PDF of all figures', value=True)
show_monthly_climate = st.sidebar.checkbox('Include Monthly Climate figure', value=True)

# =============================
# Helpers
# =============================
OUTPUT_DIR = 'wsr_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIG_REGISTRY = []  # (filename, title)


def _save_png(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def register_and_save(fig, filename, title):
    _save_png(fig, filename)
    FIG_REGISTRY.append((filename, title))


def month_formatter(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha('center')


def coerce_numeric(s):
    return pd.to_numeric(s, errors='coerce')


# Robust column getter that tolerates multiple label variants
COL_ALIASES = {
    'sample_date': ['Sample Date', 'Date', 'Sampling Date'],
    'site_id': ['Site ID: Site Name', 'Site ID', 'Station ID', 'Station'],
    'air_temp': ['Air Temp Rounded', 'Air Temperature (° C)', 'Air Temperature (°C)', 'Air Temperature (C)'],
    'water_temp': ['Water Temp Rounded', 'Water Temperature (° C)', 'Water Temperature (°C)', 'Water Temperature (C)'],
    'cond': ['Conductivity (?S/cm)', 'Conductivity (µS/cm)', 'Conductivity (uS/cm)', 'Conductivity (us/cm)'],
    'do_avg': ['Dissolved Oxygen (mg/L) Average', 'DO_avg', 'Dissolved Oxygen (mg/L)'],
    'ph': ['pH Rounded', 'pH (standard units)', 'pH'],
    'secchi': ['Secchi Disk Transparency - Average', 'Secchi', 'Secchi Disk Transparency (m)'],
    'ttube': ['Transparency Tube (meters)', 'Transparency Tube (m)', 'Transparency Tube'],
    'depth': ['Total Depth (meters)', 'Total Depth (m)', 'Total Depth']
}


def get_col(df, keys, default=None):
    for k in keys:
        if k in df.columns:
            return df[k]
    return default


# === Common styling helpers to match the provided samples (JMP-like) ===
BOXPROPS = dict(linewidth=1.5)
WHISKERPROPS = dict(linewidth=1.5)
CAPPROPS = dict(linewidth=1.5)
MEDIANPROPS = dict(linewidth=2)


def add_wqs_line(ax, y, label='WQS'):
    ax.axhline(y=y, linestyle='--', linewidth=1.5)
    # place label at left margin just above the line
    xmin, xmax = ax.get_xlim()
    ax.text(xmin - (xmax - xmin) * 0.02, y + (ax.get_ylim()[1]-ax.get_ylim()[0]) * 0.01, label)


# =============================
# Upload
# =============================
uploaded_file = st.file_uploader('Upload your Excel dataset (.xlsx)', type='xlsx')

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file)

    # ---- Prepare columns (tolerant to variant names) ----
    df = df_raw.copy()

    df['Sample Date'] = pd.to_datetime(get_col(df, COL_ALIASES['sample_date']))
    df['Site ID'] = get_col(df, COL_ALIASES['site_id']).astype(str)

    df['Air Temp Rounded'] = coerce_numeric(get_col(df, COL_ALIASES['air_temp']))
    df['Water Temp Rounded'] = coerce_numeric(get_col(df, COL_ALIASES['water_temp']))

    # Conductivity -> TDS heuristic
    cond_series = get_col(df, COL_ALIASES['cond'])
    df['Conductivity_uScm'] = coerce_numeric(cond_series)
    df['TDS (mg/L)'] = df['Conductivity_uScm'] * 0.65

    df['DO_avg'] = coerce_numeric(get_col(df, COL_ALIASES['do_avg']))
    df['pH'] = coerce_numeric(get_col(df, COL_ALIASES['ph']))
    df['Secchi'] = coerce_numeric(get_col(df, COL_ALIASES['secchi']))
    df['Transparency Tube'] = coerce_numeric(get_col(df, COL_ALIASES['ttube']))
    df['Total Depth'] = coerce_numeric(get_col(df, COL_ALIASES['depth']))

    # Drop rows with missing dates for time-based plots
    df_time = df.dropna(subset=['Sample Date'])

    # Order sites by label
    sites = [s for s in sorted(df['Site ID'].dropna().unique().tolist())]

    # =============================
    # Figure 7 (per sample): TDS boxplot by site — outline only + WQS=500
    # =============================
    fig_tds, ax = plt.subplots(figsize=(10, 6))
    data_tds = [df.loc[df['Site ID'] == s, 'TDS (mg/L)'].dropna().values for s in sites]
    ax.boxplot(
        data_tds,
        labels=sites,
        patch_artist=False,  # outline only
        boxprops=BOXPROPS,
        whiskerprops=WHISKERPROPS,
        capprops=CAPPROPS,
        medianprops=MEDIANPROPS,
        showfliers=True
    )
    add_wqs_line(ax, 500, 'WQS')
    ax.set_ylabel('Total Dissolved Solids (mg/L)')
    ax.set_xlabel('Site ID')
    ax.grid(False)
    ax.set_title('Figure 7. TDS by Site')
    register_and_save(fig_tds, 'Figure7_TDS_Boxplot.png', 'Figure 7. TDS by Site')

    # =============================
    # Figure 10: Transparency (Secchi vs. Transparency Tube) side-by-side boxplots
    # outline colored (blue/red), outliers colored accordingly, no fill
    # =============================
    fig_tr, ax = plt.subplots(figsize=(12, 6))
    methods = ['Secchi', 'Transparency Tube']
    x_positions = np.arange(len(sites))
    width = 0.35

    # Colors (will be edgecolor only; keep no fill)
    color_map = {'Secchi': 'C0', 'Transparency Tube': 'C3'}

    for i, method in enumerate(methods):
        vals = [df.loc[df['Site ID'] == s, method].dropna().values for s in sites]
        bp = ax.boxplot(
            vals,
            positions=x_positions + (i - 0.5) * width,
            widths=width * 0.9,
            patch_artist=False,
            boxprops={**BOXPROPS, 'color': color_map[method]},
            whiskerprops={**WHISKERPROPS, 'color': color_map[method]},
            capprops={**CAPPROPS, 'color': color_map[method]},
            medianprops={**MEDIANPROPS, 'color': color_map[method]},
            flierprops=dict(marker='o', markersize=4, markeredgewidth=0.8,
                            markerfacecolor=color_map[method], markeredgecolor='k'),
            manage_ticks=False,
            showfliers=True
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(sites)
    ax.set_ylabel('Transparency Tube & Secchi Disk (meters)')
    ax.set_xlabel('Site ID')
    ax.grid(False)
    ax.set_ylim(bottom=0)
    # Legend proxies
    l1, = ax.plot([], [], linestyle='-', color='C0', label='Secchi Disk')
    l2, = ax.plot([], [], linestyle='-', color='C3', label='Transparency Tube')
    ax.legend(handles=[l1, l2], loc='center right', bbox_to_anchor=(1.0, 0.5))
    ax.set_title('Figure 10. Transparency by Site')
    register_and_save(fig_tr, 'Figure10_Transparency_Boxplot.png', 'Figure 10. Transparency by Site')

    # =============================
    # Figure 6: Water Temperature — scatter by site with WQS=32.2
    # =============================
    fig_wt, ax = plt.subplots(figsize=(14, 6))
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']
    for idx, (site, g) in enumerate(df_time.groupby('Site ID')):
        ax.scatter(g['Sample Date'], g['Water Temp Rounded'], s=20,
                   marker=markers[idx % len(markers)],
                   label=site)
    add_wqs_line(ax, 32.2, 'WQS')
    ax.set_xlabel('Sample Date')
    ax.set_ylabel('Water Temperature (°C)')
    month_formatter(ax)
    ax.grid(False)
    ax.legend(title='Site ID', loc='center right', bbox_to_anchor=(1.0, 0.5))
    ax.set_title('Figure 6. Water Temperature Over Time by Site')
    register_and_save(fig_wt, 'Figure6_WaterTemperature.png', 'Figure 6. Water Temperature Over Time by Site')

    # =============================
    # Figure 8: Dissolved Oxygen — outline boxplot with WQS=5
    # =============================
    fig_do, ax = plt.subplots(figsize=(10, 6))
    data_do = [df.loc[df['Site ID'] == s, 'DO_avg'].dropna().values for s in sites]
    ax.boxplot(
        data_do,
        labels=sites,
        patch_artist=False,
        boxprops=BOXPROPS,
        whiskerprops=WHISKERPROPS,
        capprops=CAPPROPS,
        medianprops=MEDIANPROPS,
        showfliers=True
    )
    add_wqs_line(ax, 5.0, 'WQS')
    ax.set_ylabel('Dissolved Oxygen (mg/L)')
    ax.set_xlabel('Site ID')
    ax.grid(False)
    ax.set_title('Figure 8. Dissolved Oxygen by Site')
    register_and_save(fig_do, 'Figure8_DO_Boxplot.png', 'Figure 8. Dissolved Oxygen by Site')

    # =============================
    # Figure 9: pH — outline boxplot with two reference lines (6.5, 9.0)
    # =============================
    fig_ph, ax = plt.subplots(figsize=(10, 6))
    data_ph = [df.loc[df['Site ID'] == s, 'pH'].dropna().values for s in sites]
    ax.boxplot(
        data_ph,
        labels=sites,
        patch_artist=False,
        boxprops=BOXPROPS,
        whiskerprops=WHISKERPROPS,
        capprops=CAPPROPS,
        medianprops=MEDIANPROPS,
        showfliers=True
    )
    # Add WQS Min/Max like samples
    ymin, ymax = 6.5, 9.0
    ax.axhline(y=ymax, linestyle='--', linewidth=1.5)
    ax.axhline(y=ymin, linestyle='--', linewidth=1.5)
    xmin, xmax = ax.get_xlim()
    ax.text(xmin - (xmax - xmin) * 0.02, ymax + 0.03*(ax.get_ylim()[1]-ax.get_ylim()[0]), 'WQS Max')
    ax.text(xmin - (xmax - xmin) * 0.02, ymin - 0.06*(ax.get_ylim()[1]-ax.get_ylim()[0]), 'WQS Min')
    ax.set_ylabel('pH (standard units)')
    ax.set_xlabel('Site ID')
    ax.grid(False)
    ax.set_title('Figure 9. pH by Site')
    register_and_save(fig_ph, 'Figure9_pH_Boxplot.png', 'Figure 9. pH by Site')

    # =============================
    # Figure 11: Total Depth — outline boxplot
    # =============================
    fig_depth, ax = plt.subplots(figsize=(10, 6))
    data_depth = [df.loc[df['Site ID'] == s, 'Total Depth'].dropna().values for s in sites]
    ax.boxplot(
        data_depth,
        labels=sites,
        patch_artist=False,
        boxprops=BOXPROPS,
        whiskerprops=WHISKERPROPS,
        capprops=CAPPROPS,
        medianprops=MEDIANPROPS,
        showfliers=True
    )
    ax.set_ylabel('Total Depth (m)')
    ax.set_xlabel('Site ID')
    ax.grid(False)
    ax.set_title('Figure 11. Total Depth by Site')
    register_and_save(fig_depth, 'Figure11_TotalDepth_Boxplot.png', 'Figure 11. Total Depth by Site')

    # =============================
    # Monthly climate (optional; kept minimal styling)
    # =============================
    if show_monthly_climate:
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        precipitation = [2.2, 3.2, 3.9, 4.3, 5.3, 4.1, 2.6, 2.8, 3.4, 4.6, 3.3, 3.0]
        temperature = [7.2, 9.5, 13.8, 18.2, 23.3, 27.8, 29.7, 29.4, 25.1, 18.9, 12.4, 8.4]
        figc, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Temperature (°C)')
        ax1.plot(months, temperature, linewidth=2)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Precipitation (inches)')
        ax2.bar(months, precipitation, alpha=0.7)
        ax1.set_title('Figure 12. Monthly Avg Precipitation and Temperature')
        register_and_save(figc, 'Figure12_MonthlyClimate.png', 'Figure 12. Monthly Avg Precipitation and Temperature')

    # =============================
    # Table 6 — Summary Statistics by Site and Parameter
    # =============================
    st.markdown('## 📋 Table 6. Summary Statistics by Site and Parameter')

    site_event_counts = df.groupby('Site ID').size()
    valid_sites = site_event_counts[site_event_counts >= min_events_per_site].index
    df_valid = df[df['Site ID'].isin(valid_sites)]

    param_map = {
        'Air Temperature (°C)': 'Air Temp Rounded',
        'Water Temperature (°C)': 'Water Temp Rounded',
        'Dissolved Oxygen (mg/L)': 'DO_avg',
        'pH (standard units)': 'pH',
        'TDS (mg/L)': 'TDS (mg/L)',
        'Secchi Disk Transparency (m)': 'Secchi',
        'Transparency Tube (m)': 'Transparency Tube',
        'Total Depth (m)': 'Total Depth',
    }

    rows = []
    for label, column in param_map.items():
        for stat in ['Mean', 'Std Dev', 'Range']:
            row = {'Parameter': label if stat == 'Mean' else '', 'Statistic': stat}
            for site in valid_sites:
                values = df_valid.loc[df_valid['Site ID'] == site, column].dropna()
                if len(values) >= min_events_per_site:
                    if stat == 'Mean':
                        val = round(values.mean(), 2)
                    elif stat == 'Std Dev':
                        val = round(values.std(ddof=1), 2)
                    else:
                        val = round(values.max() - values.min(), 2) if not values.empty else np.nan
                else:
                    val = 'ND'
                row[site] = val
            rows.append(row)

    table6_df = pd.DataFrame(rows)

    def style_word_format(df_in: pd.DataFrame) -> pd.DataFrame:
        styled = df_in.copy()
        styled['Parameter'] = styled['Parameter'].mask(styled['Parameter'] == '').ffill().mask(styled['Statistic'] != 'Mean', '')
        return styled

    table6_formatted = style_word_format(table6_df)

    # Save CSV + Excel
    csv_path = os.path.join(OUTPUT_DIR, 'Table6_Summary.csv')
    xlsx_path = os.path.join(OUTPUT_DIR, 'Table6_Summary.xlsx')
    table6_df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        table6_formatted.to_excel(writer, sheet_name='Table6', index=False)
        ws = writer.sheets['Table6']
        for i, col in enumerate(table6_formatted.columns, 1):
            ws.set_column(i-1, i-1, max(12, len(str(col)) + 2))

    st.dataframe(table6_formatted, use_container_width=True)

    # =============================
    # Multi-page PDF (optional)
    # =============================
    pdf_bytes = None
    if make_pdf and len(FIG_REGISTRY) > 0:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            for (filename, title) in FIG_REGISTRY:
                img = plt.imread(os.path.join(OUTPUT_DIR, filename))
                h, w = img.shape[:2]
                dpi = 100
                fig_pdf = plt.figure(figsize=(w/dpi, h/dpi))
                plt.imshow(img)
                plt.axis('off')
                plt.title(title)
                pdf.savefig(fig_pdf, bbox_inches='tight')
                plt.close(fig_pdf)
        pdf_bytes = pdf_buf.getvalue()
        with open(os.path.join(OUTPUT_DIR, 'WSR_Figures.pdf'), 'wb') as f:
            f.write(pdf_bytes)

    # =============================
    # ZIP download (figures + Table6 + PDF)
    # =============================
    st.markdown('## 📦 Download All Results (Figures + Table 6)')
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(OUTPUT_DIR):
            filepath = os.path.join(OUTPUT_DIR, file)
            zipf.write(filepath, arcname=file)
    zip_buffer.seek(0)
    st.download_button('📥 Download ZIP', data=zip_buffer, file_name='WSR_All_Results.zip', mime='application/zip')

    st.success('✅ All outputs successfully generated.')

    # =============================
    # On-screen preview
    # =============================
    for (filename, title) in FIG_REGISTRY:
        st.subheader(title)
        img = plt.imread(os.path.join(OUTPUT_DIR, filename))
        st.image(img, use_column_width=True)

    if pdf_bytes:
        st.download_button('📄 Download multi-page PDF', data=pdf_bytes, file_name='WSR_Figures.pdf', mime='application/pdf')
