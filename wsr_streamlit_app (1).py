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

    # --- Prepare columns for calculations ---
    df['Sample Date'] = pd.to_datetime(df['Sample Date'])
    df['Site ID'] = df['Site ID: Site Name'].astype(str)

    # Create consistent/expected columns for Table 6
    df['Air Temp Rounded'] = pd.to_numeric(df['Air Temperature (° C)'], errors='coerce')
    df['DO_avg'] = pd.to_numeric(df['Dissolved Oxygen (mg/L) Average'], errors='coerce')
    df['Conductivity'] = pd.to_numeric(df['Conductivity (?S/cm)'], errors='coerce')
    df['TDS (mg/L)'] = df['Conductivity'] * 0.65
    df['Water Temp Rounded'] = pd.to_numeric(df.get('Water Temp Rounded'), errors='coerce')
    df['pH'] = pd.to_numeric(df.get('pH Rounded'), errors='coerce')
    df['Secchi'] = pd.to_numeric(df.get('Secchi Disk Transparency - Average'), errors='coerce')
    df['Transparency Tube'] = pd.to_numeric(df.get('Transparency Tube (meters)'), errors='coerce')
    df['Total Depth'] = pd.to_numeric(df.get('Total Depth (meters)'), errors='coerce')

    output_dir = "wsr_figures"
    os.makedirs(output_dir, exist_ok=True)

    def save_figure(fig, filename):
        fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # --- Figure 6: Water Temp
    fig6, ax = plt.subplots(figsize=(14, 6))
    sns.scatterplot(data=df, x='Sample Date', y='Water Temp Rounded', hue='Site ID', s=40, ax=ax)
    ax.axhline(y=32.2, color='red', linestyle='--')
    ax.text(df['Sample Date'].min(), 32.6, 'WQS = 32.2°C', color='red')
    ax.set_xlabel('Sample Date')
    ax.set_ylabel('Water Temperature (°C)')
    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="Site ID")
    save_figure(fig6, "Figure6_WaterTemperature.png")

    # --- Figure 7: TDS
    fig7, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Site ID', y='TDS (mg/L)', color='white', fliersize=4, ax=ax)
    ax.axhline(y=500, color='red', linestyle='--')
    ax.text(-0.4, 510, 'WQS = 500 mg/L', color='red')
    ax.set_ylabel('TDS (mg/L)')
    save_figure(fig7, "Figure7_TDS_Boxplot.png")

    # --- Figure 8: DO
    fig8, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Site ID', y='DO_avg', color='white', fliersize=4, ax=ax)
    ax.axhline(y=5.0, color='red', linestyle='--')
    ax.text(-0.4, 5.1, 'WQS = 5.0 mg/L', color='red')
    ax.set_ylabel('Dissolved Oxygen (mg/L)')
    save_figure(fig8, "Figure8_DO_Boxplot.png")

    # --- Figure 10: Transparency
    transparency_df = df.melt(
        id_vars=['Site ID'],
        value_vars=['Secchi', 'Transparency Tube'],
        var_name='Transparency Type',
        value_name='Transparency (m)'
    )
    fig10, ax = plt.subplots(figsize=(12, 6))
    palette = {'Secchi': 'blue', 'Transparency Tube': 'red'}
    sns.boxplot(
        data=transparency_df,
        x='Site ID',
        y='Transparency (m)',
        hue='Transparency Type',
        ax=ax,
        palette=palette,
        linewidth=2,
        fliersize=0
    )
    ax.set_ylabel('Transparency (meters)')
    ax.set_ylim(0, 0.7)
    ax.set_title("Figure 10. Transparency by Site")
    ax.legend(title="Method", loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig10.tight_layout()
    save_figure(fig10, "Figure10_Transparency_Boxplot.png")

    # --- Figure 11: Total Depth
    fig11, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Site ID', y='Total Depth', color='white', fliersize=4, ax=ax)
    ax.set_ylabel('Total Depth (m)')
    save_figure(fig11, "Figure11_TotalDepth_Boxplot.png")

    # --- Monthly Climate Chart
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    precipitation = [2.2, 3.2, 3.9, 4.3, 5.3, 4.1, 2.6, 2.8, 3.4, 4.6, 3.3, 3.0]
    temperature = [7.2, 9.5, 13.8, 18.2, 23.3, 27.8, 29.7, 29.4, 25.1, 18.9, 12.4, 8.4]
    fig_climate, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Temperature (°C)', color='red')
    ax1.plot(months, temperature, color='red', linewidth=3)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Precipitation (inches)', color='blue')
    ax2.bar(months, precipitation, color='blue', alpha=0.7)
    fig_climate.tight_layout()
    save_figure(fig_climate, "Figure_MonthlyClimate.png")

    # --- Table 6: Summary Statistics
    st.markdown("## 📋 Table 6. Summary Statistics by Site and Parameter")

    site_event_counts = df.groupby('Site ID').size()
    valid_sites = site_event_counts[site_event_counts >= 10].index
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

    summary_rows = []
    for label, column in param_map.items():
        for stat in ['Mean', 'Std Dev', 'Range']:
            row = {'Parameter': label, 'Statistic': stat}
            for site in valid_sites:
                values = df_valid[df_valid['Site ID'] == site][column].dropna()
                if len(values) >= 10:
                    if stat == 'Mean':
                        val = round(values.mean(), 2)
                    elif stat == 'Std Dev':
                        val = round(values.std(), 2)
                    elif stat == 'Range':
                        val = round(values.max() - values.min(), 2)
                else:
                    val = 'ND'
                row[site] = val
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save Table 6 as CSV
    table6_path = os.path.join(output_dir, "Table6_Summary.csv")
    summary_df.to_csv(table6_path, index=False)
    st.dataframe(summary_df.style.set_properties(**{'text-align': 'center'}), use_container_width=True)

    # --- ZIP creation with all figures + table
    st.markdown("## 📦 Download All Results (Figures + Table 6)")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for file in os.listdir(output_dir):
            filepath = os.path.join(output_dir, file)
            zipf.write(filepath, arcname=file)
    zip_buffer.seek(0)
    st.download_button("📥 Download ZIP", data=zip_buffer, file_name="WSR_All_Results.zip", mime="application/zip")
    st.success("✅ All outputs successfully generated.")

    # --- Show all charts
    st.subheader("Figure 6. Water Temperature Over Time by Site")
    st.pyplot(fig6)

    st.subheader("Figure 7. TDS by Site")
    st.pyplot(fig7)

    st.subheader("Figure 8. Dissolved Oxygen by Site")
    st.pyplot(fig8)

    st.subheader("Figure 10. Transparency by Site")
    st.pyplot(fig10)

    st.subheader("Figure 11. Total Depth by Site")
    st.pyplot(fig11)

    st.subheader("Figure #: Monthly Avg Precipitation and Temperature")
    st.pyplot(fig_climate)
