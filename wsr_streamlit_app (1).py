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

    # --- Figure 10: Transparency with Full Custom Coloring ---
    transparency_df = df.melt(
        id_vars=['Site ID'],
        value_vars=['Secchi', 'Transparency Tube'],
        var_name='Transparency Type',
        value_name='Transparency (m)'
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = {'Secchi': 'blue', 'Transparency Tube': 'red'}

    # ترسیم اولیه بدون نمایش آوتلایر
    sns_plot = sns.boxplot(
        data=transparency_df,
        x='Site ID',
        y='Transparency (m)',
        hue='Transparency Type',
        ax=ax,
        palette=palette,
        linewidth=2,
        fliersize=0  # آوتلایرها رو دستی می‌سازیم
    )

    # رنگی کردن outline، خط میانی، whisker و cap به رنگ پارامتر
    for i, (artist, line) in enumerate(zip(ax.artists, ax.lines[::6])):
        param_index = i % 2
        param = list(palette.keys())[param_index]
        color = palette[param]
        artist.set_facecolor('white')
        artist.set_edgecolor(color)
        artist.set_linewidth(2)

        # خطوط: هر boxplot 6 خط داره: [median, whisker low, whisker high, cap low, cap high, hidden]
        for j in range(6):
            line_index = i * 6 + j
            ax.lines[line_index].set_color(color)
            ax.lines[line_index].set_linewidth(2)

    # رسم دستی آوتلایرها با رنگ مربوط به پارامتر
    grouped = transparency_df.groupby(['Site ID', 'Transparency Type'])
    for (site, param), group in grouped:
        pos = list(df['Site ID'].unique()).index(site)
        shift = -0.2 if param == 'Secchi' else 0.2
        x_val = pos + shift
        values = group['Transparency (m)'].dropna()

        # محاسبه IQR برای شناسایی آوتلایر
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = values[(values < lower) | (values > upper)]

        ax.scatter(
            [x_val] * len(outliers),
            outliers,
            color=palette[param],
            s=30,
            edgecolors='k',
            linewidths=0.5,
            alpha=0.8,
            zorder=10
        )

    # تنظیمات نهایی
    ax.set_ylabel('Transparency (meters)')
    ax.set_ylim(0, 0.7)
    ax.set_title("Figure 10. Transparency by Site")
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
