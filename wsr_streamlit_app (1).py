import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
from io import BytesIO

st.set_page_config(page_title='WSR Graph Generator', layout='wide')
st.title("📊 Watershed Summary Report Graph Generator")

uploaded_file = st.file_uploader("Upload your Excel dataset (.xlsx)", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # --- Prepare columns ---
    df['Sample Date'] = pd.to_datetime(df['Sample Date'])
    df['Site ID'] = df['Site ID: Site Name'].astype(str)
    df['Water Temp Rounded'] = pd.to_numeric(df['Water Temp Rounded'], errors='coerce')
    df['Conductivity'] = pd.to_numeric(df['Conductivity (?S/cm)'], errors='coerce')
    df['TDS (mg/L)'] = df['Conductivity'] * 0.65
    df['DO1'] = pd.to_numeric(df['DO1 Rounded'], errors='coerce')
    df['DO2'] = pd.to_numeric(df['DO2 Rounded'], errors='coerce')
    df['DO_avg'] = df[['DO1', 'DO2']].mean(axis=1)
    df['pH'] = pd.to_numeric(df['pH Rounded'], errors='coerce')
    df['Secchi'] = pd.to_numeric(df['Secchi Disk Transparency - Average'], errors='coerce')
    df['Transparency Tube'] = pd.to_numeric(df['Transparency Tube (meters)'], errors='coerce')
    df['Total Depth'] = pd.to_numeric(df['Total Depth (meters)'], errors='coerce')

    output_dir = "wsr_figures"
    os.makedirs(output_dir, exist_ok=True)

    def save_figure(fig, filename):
        fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # --- Figure 6: Water Temp ---
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.scatterplot(data=df, x='Sample Date', y='Water Temp Rounded', hue='Site ID', s=40, ax=ax)
    ax.axhline(y=32.2, color='red', linestyle='--')
    ax.text(df['Sample Date'].min(), 32.6, 'WQS = 32.2°C', color='red')
    ax.set_xlabel('Sample Date')
    ax.set_ylabel('Water Temperature (°C)')
    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="Site ID")
    save_figure(fig, "Figure6_WaterTemperature.png")
    st.subheader("Figure 6. Water Temperature Over Time by Site")
    st.pyplot(fig)

    # --- Figure 7: TDS ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Site ID', y='TDS (mg/L)', color='white', fliersize=4, ax=ax)
    ax.axhline(y=500, color='red', linestyle='--')
    ax.text(-0.4, 510, 'WQS = 500 mg/L', color='red')
    ax.set_ylabel('TDS (mg/L)')
    save_figure(fig, "Figure7_TDS_Boxplot.png")
    st.subheader("Figure 7. TDS by Site")
    st.pyplot(fig)

    # --- Figure 8: DO ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Site ID', y='DO_avg', color='white', fliersize=4, ax=ax)
    ax.axhline(y=5.0, color='red', linestyle='--')
    ax.text(-0.4, 5.1, 'WQS = 5.0 mg/L', color='red')
    ax.set_ylabel('Dissolved Oxygen (mg/L)')
    save_figure(fig, "Figure8_DO_Boxplot.png")
    st.subheader("Figure 8. Dissolved Oxygen by Site")
    st.pyplot(fig)

    # --- Figure 9: pH ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Site ID', y='pH', color='white', fliersize=4, ax=ax)
    ax.axhline(y=9.0, color='red', linestyle='--')
    ax.axhline(y=6.5, color='red', linestyle='--')
    ax.text(-0.4, 9.1, 'WQS Max = 9.0', color='red')
    ax.text(-0.4, 6.6, 'WQS Min = 6.5', color='red')
    ax.set_ylabel('pH (standard units)')
    save_figure(fig, "Figure9_pH_Boxplot.png")
    st.subheader("Figure 9. pH by Site")
    st.pyplot(fig)

    # --- Figure 10: Transparency - Outline Color Only (No Fill), Colored Outliers ---
    transparency_df = df.melt(
        id_vars=['Site ID'],
        value_vars=['Secchi', 'Transparency Tube'],
        var_name='Transparency Type',
        value_name='Transparency (m)'
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    palette = {'Secchi': 'blue', 'Transparency Tube': 'red'}

    # رسم باکس‌پلات‌ها
    sns.boxplot(
        data=transparency_df,
        x='Site ID',
        y='Transparency (m)',
        hue='Transparency Type',
        ax=ax,
        palette=palette,
        linewidth=2,
        fliersize=5,
        boxprops=dict(facecolor='none'),  # بدون رنگ داخلی
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='o', markersize=5, linestyle='none')  # رنگش رو دستی تنظیم می‌کنیم
    )

    # تنظیم رنگ خط دور باکس‌ها برای هر پارامتر
    for i, artist in enumerate(ax.artists):
        method = list(palette.keys())[i % 2]  # Secchi یا Tube
        color = palette[method]
        artist.set_edgecolor(color)
        artist.set_facecolor('none')
        artist.set_linewidth(2)

    # تنظیم رنگ نقاط بیرونی (outliers)
    flier_lines = [line for idx, line in enumerate(ax.lines) if idx % 6 in [4, 5]]  # flier lines
    for idx, line in enumerate(flier_lines):
        method = list(palette.keys())[idx % 2]
        line.set_color(palette[method])

    ax.set_ylabel('Transparency Tube & Secchi Disk (meters)')
    ax.set_title("Figure 10. Transparency by Site")
    ax.set_ylim(0, 0.7)
    ax.legend(title="Method", loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()

    save_figure(fig, "Figure10_Transparency_Boxplot.png")
    st.subheader("Figure 10. Transparency by Site")
    st.pyplot(fig)


    # --- Figure 11: Total Depth ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Site ID', y='Total Depth', color='white', fliersize=4, ax=ax)
    ax.set_ylabel('Total Depth (m)')
    save_figure(fig, "Figure11_TotalDepth_Boxplot.png")
    st.subheader("Figure 11. Total Depth by Site")
    st.pyplot(fig)

    # --- Climate chart ---
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    precipitation = [2.2, 3.2, 3.9, 4.3, 5.3, 4.1, 2.6, 2.8, 3.4, 4.6, 3.3, 3.0]
    temperature = [7.2, 9.5, 13.8, 18.2, 23.3, 27.8, 29.7, 29.4, 25.1, 18.9, 12.4, 8.4]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Temperature (°C)', color='red')
    ax1.plot(months, temperature, color='red', linewidth=3)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Precipitation (inches)', color='blue')
    ax2.bar(months, precipitation, color='blue', alpha=0.7)
    fig.tight_layout()
    save_figure(fig, "Figure_MonthlyClimate.png")
    st.subheader("Figure #: Monthly Avg Precipitation and Temperature in Denton County")
    st.pyplot(fig)

    # --- Create ZIP for download ---
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for file in os.listdir(output_dir):
            zipf.write(os.path.join(output_dir, file), arcname=file)
    zip_buffer.seek(0)

    st.success("✅ All figures generated successfully.")
    st.download_button("📥 Download All Graphs (ZIP)", data=zip_buffer, file_name="WSR_Graphs.zip", mime="application/zip")
