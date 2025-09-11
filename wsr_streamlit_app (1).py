# ================== Summary Table (Table 6 style) ==================
param_map = {
    'Air Temp Rounded': 'Air Temperature (°C)',
    'Water Temp Rounded': 'Water Temperature (°C)',
    'DO_avg': 'Dissolved Oxygen (mg/L)',
    'pH': 'pH (standard units)',
    'Conductivity': 'Conductivity (µS/cm)',
    'Secchi': 'Secchi Disk Transparency (m)',
    'Transparency Tube': 'Transparency Tube (m)',
    'Total Depth': 'Total Depth (m)',
    'TDS (mg/L)': 'Total Dissolved Solids (mg/L)',
    'E_coli': 'E. coli (#/100 mL)',
}

summary_rows = []
for col, pname in param_map.items():
    if col in df.columns and df[col].notna().any():
        for stat in ['Mean', 'Std Dev', 'Range']:
            row = {'Parameter': pname, 'Statistic': stat}
            for s in site_order:
                vals = df.loc[df['Site ID'].eq(s), col].dropna().values
                if len(vals) == 0:
                    row[s] = "ND"
                else:
                    if stat == 'Mean':
                        row[s] = round(np.mean(vals), 1)
                    elif stat == 'Std Dev':
                        row[s] = round(np.std(vals, ddof=1), 1)
                    elif stat == 'Range':
                        row[s] = round(np.max(vals) - np.min(vals), 1)
            summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

# نمایش در اپ
st.subheader("📑 Summary Statistics (Table 6 style)")
st.dataframe(summary_df)

# ذخیره در Excel و اضافه به ZIP
table6_path = os.path.join(output_dir, "Table6_Summary.xlsx")
summary_df.to_excel(table6_path, index=False)
