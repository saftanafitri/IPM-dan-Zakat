import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi IPM Berbasis Zakat",
    page_icon="💸",
    layout="wide"
)

# --- FUNGSI-FUNGSI ---

# Cache resource agar model tidak di-load ulang setiap kali ada interaksi
@st.cache_resource
def load_model_and_components():
    """Memuat model dan semua komponen preprocessing yang tersimpan."""
    try:
        components = joblib.load('model_skema7H.pkl')
        return components
    except FileNotFoundError:
        st.error("File 'model_skema7H.pkl' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Cache data agar tidak di-load ulang
@st.cache_data
def load_raw_data():
    """Memuat data mentah untuk EDA."""
    try:
        df = pd.read_excel("bps-od_15042_indeks_pmbngnn_manusia__prov_di_indonesia_data.xlsx")
        df['zakat_per_kapita'] = df['total_zakat'] / df['penduduk']
        return df
    except FileNotFoundError:
        st.warning("File data mentah tidak ditemukan, bagian EDA tidak akan ditampilkan.")
        return None
    except Exception as e:
        st.error(f"Gagal memuat data mentah: {e}")
        return None

def predict_ipm(input_data, components):
    """Fungsi untuk melakukan prediksi IPM berdasarkan input pengguna."""
    model = components['model']
    pt = components['power_transformer']
    scaler = components['scaler']
    features = components['features']
    transform_features = components['transform_features']
    province_mapping = components['province_mapping']
    random_effects = components['random_effects']
    tahun_min = components['tahun_min']
    
    # Menyiapkan data input
    input_data['zakat_per_kapita'] = input_data['total_zakat'] / input_data['penduduk']
    input_data['tahun_num'] = input_data['tahun'] - tahun_min
    
    input_df = pd.DataFrame([input_data])
    
    # Transformasi & Scaling
    try:
        input_df[transform_features] = pt.transform(input_df[transform_features])
        input_df[features] = scaler.transform(input_df[features])
    except Exception as e:
        st.error(f"Error saat preprocessing data: {e}")
        return None, None, None, None

    provinsi_input = input_data['provinsi'].upper()
    kode_provinsi = province_mapping.get(provinsi_input, -1)

    if kode_provinsi == -1:
        st.error("Provinsi tidak ditemukan dalam data training.")
        return None, None, None, None

    # Prediksi Fixed Effect
    fe_params = model.fe_params
    X = input_df[features]
    fixed_effect = fe_params['Intercept'] + np.dot(X, fe_params[features].values)[0]

    # Analisis Kontribusi
    bidang_list = ['pendidikan', 'kesehatan', 'ekonomi', 'kemanusiaan']
    bidang_kontribusi = {}
    title_mapping = {
        'pendidikan': 'Pendidikan', 'kesehatan': 'Kesehatan',
        'ekonomi': 'Ekonomi', 'kemanusiaan': 'Kemanusiaan'
    }

    for bidang in bidang_list:
        fix = fe_params.get(bidang, 0) * input_df[bidang].values[0]
        rand = 0
        if kode_provinsi in random_effects and bidang in random_effects[kode_provinsi]:
            rand = random_effects[kode_provinsi][bidang] * input_df[bidang].values[0]
        total = fix + rand
        bidang_kontribusi[bidang] = total

    # Efek Acak (Random Effect)
    random_intercept = 0
    if kode_provinsi in random_effects:
        random_intercept = random_effects[kode_provinsi].get('Group', 0)

    prediction = fixed_effect + random_intercept

    # Klasifikasi kontribusi
    signifikan = {}
    tidak_signifikan = []
    
    for bidang in bidang_list:
        nilai = bidang_kontribusi[bidang]
        if nilai > 0.001:  # Hanya ambil kontribusi positif yang cukup berarti
            signifikan[title_mapping[bidang]] = nilai
        else:
            tidak_signifikan.append(title_mapping[bidang])
            
    total_signifikan = sum(signifikan.values())
    persentase_signifikan = {bidang: (nilai / total_signifikan) * 100 if total_signifikan > 0 else 0 for bidang, nilai in signifikan.items()}

    return prediction, signifikan, persentase_signifikan, tidak_signifikan

# --- TAMPILAN APLIKASI (UI) ---

# Judul Utama
st.title("💸 Analisis Pengaruh Zakat terhadap Indeks Pembangunan Manusia (IPM)")
st.markdown("Aplikasi interaktif untuk memprediksi IPM di berbagai provinsi di Indonesia berdasarkan alokasi dana zakat, menggunakan *Mixed Effects Model*.")

# Memuat komponen model
components = load_model_and_components()
df_raw = load_raw_data()

if components and df_raw is not None:
    # --- Kolom Sidebar ---
    st.sidebar.header("Tentang Proyek")
    st.sidebar.info(
        "Proyek ini menganalisis data Indeks Pembangunan Manusia (IPM) dan data pendistribusian "
        "zakat dari Badan Pusat Statistik (BPS) dan sumber data terbuka lainnya. "
        "Model yang digunakan adalah **Mixed-Effects Regression** untuk memperhitungkan variasi "
        "antar provinsi (random effects)."
    )
    st.sidebar.header("Parameter Input")
    
    # Input dari Pengguna di Sidebar
    provinsi_list = sorted(df_raw['nama_provinsi'].unique())
    provinsi = st.sidebar.selectbox("Pilih Provinsi:", provinsi_list, index=provinsi_list.index('JAWA BARAT'))
    
    tahun = st.sidebar.number_input("Tahun Prediksi:", min_value=2024, max_value=2030, value=2025)
    
    penduduk = st.sidebar.number_input("Jumlah Penduduk (jiwa):", min_value=100000, value=50000000, step=100000)
    
    total_zakat = st.sidebar.number_input("Total Pengumpulan Zakat (Rp):", min_value=1000000000, value=50000000000, step=1000000000, format="%d")
    
    st.sidebar.subheader("Alokasi Dana Zakat (%)")
    
    persen_pendidikan = st.sidebar.slider("Pendidikan (%)", 0, 100, 30)
    persen_kesehatan = st.sidebar.slider("Kesehatan (%)", 0, 100, 20)
    persen_ekonomi = st.sidebar.slider("Ekonomi (%)", 0, 100, 40)
    
    # Persentase sisanya dialokasikan ke Kemanusiaan
    sisa_persen = 100 - (persen_pendidikan + persen_kesehatan + persen_ekonomi)
    if sisa_persen < 0:
        st.sidebar.error("Total persentase alokasi tidak boleh melebihi 100%.")
        persen_kemanusiaan = 0
    else:
        persen_kemanusiaan = sisa_persen
    
    st.sidebar.metric("Sisa untuk Kemanusiaan (%)", f"{persen_kemanusiaan}%")

    # Tombol Prediksi
    predict_button = st.sidebar.button("Prediksi IPM", use_container_width=True)

    # --- Tampilan Utama ---

    tab1, tab2, tab3 = st.tabs(["📊 Dasbor Prediksi", "📈 Visualisasi Data (EDA)", "🧠 Penjelasan Model"])

    with tab1:
        st.header("Hasil Prediksi IPM")
        if predict_button:
            if sisa_persen < 0:
                st.error("Gagal melakukan prediksi. Mohon perbaiki total persentase alokasi di sidebar.")
            else:
                user_input = {
                    'provinsi': provinsi,
                    'tahun': tahun,
                    'total_zakat': total_zakat,
                    'kemanusiaan': total_zakat * (persen_kemanusiaan / 100),
                    'kesehatan': total_zakat * (persen_kesehatan / 100),
                    'pendidikan': total_zakat * (persen_pendidikan / 100),
                    'ekonomi': total_zakat * (persen_ekonomi / 100),
                    'penduduk': penduduk
                }
                
                ipm_pred, kontribusi_poin, kontribusi_persen, tidak_signifikan = predict_ipm(user_input, components)

                if ipm_pred is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label=f"Prediksi IPM untuk {provinsi} ({tahun})",
                            value=f"{ipm_pred:.2f}"
                        )
                    with col2:
                        zpk = user_input['total_zakat'] / user_input['penduduk']
                        st.metric(
                            label="Zakat per Kapita",
                            value=f"Rp {zpk:,.0f}"
                        )

                    st.markdown("---")
                    st.subheader("Analisis Kontribusi Program Zakat")
                    
                    if kontribusi_poin:
                        sorted_kontribusi = sorted(kontribusi_poin.items(), key=lambda x: x[1], reverse=True)
                        
                        df_kontribusi = pd.DataFrame(sorted_kontribusi, columns=['Program', 'Kontribusi Poin IPM'])
                        df_kontribusi['Kontribusi (%)'] = df_kontribusi['Program'].map(kontribusi_persen)
                        
                        st.dataframe(
                            df_kontribusi,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Kontribusi Poin IPM": st.column_config.ProgressColumn(
                                    "Kontribusi Poin IPM",
                                    format="%.4f",
                                    min_value=0,
                                    max_value=float(df_kontribusi['Kontribusi Poin IPM'].max())
                                ),
                                "Kontribusi (%)": st.column_config.ProgressColumn(
                                    "Kontribusi (%)",
                                    format="%.2f%%",
                                    min_value=0,
                                    max_value=100
                                )
                            }
                        )

                        total_kontribusi_positif = sum(kontribusi_poin.values())
                        st.info(f"**Total Kontribusi Positif:** Prediksi kenaikan **{total_kontribusi_positif:.4f}** poin IPM dari program zakat yang signifikan.")
                        
                        st.subheader("💡 Saran Strategis")
                        bidang_terbesar = sorted_kontribusi[0][0]
                        st.success(f"**Fokus Utama:** Alokasi dana pada bidang **{bidang_terbesar}** memberikan dampak paling signifikan terhadap peningkatan IPM di provinsi ini.")

                    else:
                        st.warning("Berdasarkan input yang diberikan, tidak ada program zakat yang memberikan kontribusi positif signifikan terhadap IPM.")

                    if tidak_signifikan:
                        st.warning(f"**Perlu Evaluasi:** Program di bidang **{', '.join(tidak_signifikan)}** tidak menunjukkan kontribusi signifikan atau bahkan berpotensi menurunkan IPM. Perlu tinjauan ulang strategi pada bidang ini.")

        else:
            st.info("Masukkan parameter di sidebar kiri dan klik tombol 'Prediksi IPM' untuk melihat hasilnya.")

    with tab2:
        st.header("Eksplorasi Data Awal (EDA)")
        
        # EDA 1: Top 5 Zakat
        st.subheader("Top 5 Provinsi dengan Total Zakat Tertinggi (2021-2024)")
        top_zakat = df_raw.groupby('nama_provinsi')['total_zakat'].sum().nlargest(5)
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_zakat.values / 1e9, y=top_zakat.index, palette="viridis", ax=ax1)
        ax1.set_title("Top 5 Provinsi dengan Total Zakat Tertinggi")
        ax1.set_xlabel("Total Zakat (Miliar Rupiah)")
        ax1.set_ylabel("Provinsi")
        st.pyplot(fig1)
        
        # EDA 2: Top 5 IPM
        st.subheader("Top 5 Provinsi dengan IPM Tertinggi (Rata-rata 2021-2024)")
        top_ipm = df_raw.groupby('nama_provinsi')['indeks_pembangunan_manusia'].mean().nlargest(5)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_ipm.values, y=top_ipm.index, palette="magma", ax=ax2)
        ax2.set_title("Top 5 Provinsi dengan IPM Tertinggi")
        ax2.set_xlabel("Indeks Pembangunan Manusia (IPM)")
        ax2.set_ylabel("Provinsi")
        st.pyplot(fig2)

        # EDA 3: Zakat Per Kapita
        st.subheader("Top 5 Provinsi dengan Zakat per Kapita Tertinggi (Rata-rata 2021-2024)")
        top_zpk = df_raw.groupby('nama_provinsi')['zakat_per_kapita'].mean().nlargest(5)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_zpk.values, y=top_zpk.index, palette='crest', ax=ax3)
        ax3.set_title("Top 5 Provinsi dengan Zakat Per Kapita Tertinggi")
        ax3.set_xlabel("Zakat per Kapita (Rupiah)")
        ax3.set_ylabel("Provinsi")
        st.pyplot(fig3)

    with tab3:
        st.header("Penjelasan Model dan Metodologi")
        st.markdown("""
        Model yang digunakan dalam analisis ini adalah **Mixed-Effects Linear Model (Model Efek Campuran)**. Model ini dipilih karena kemampuannya untuk menangani data yang memiliki struktur hirarkis atau berkelompok, seperti data dari berbagai provinsi selama beberapa tahun.

        #### Mengapa Mixed-Effects Model?
        Data IPM dan zakat dari setiap provinsi cenderung memiliki korelasi internal. Artinya, pengukuran dari provinsi yang sama (misalnya IPM Jawa Barat tahun 2021 dan 2022) lebih mirip satu sama lain dibandingkan dengan pengukuran dari provinsi lain. Model regresi linear biasa tidak dapat menangani korelasi ini, yang dapat menyebabkan kesimpulan yang salah.

        **Intraclass Correlation Coefficient (ICC)** dari data ini adalah **0.889**, yang menunjukkan bahwa 88.9% variasi IPM disebabkan oleh perbedaan antar provinsi. Nilai ICC yang tinggi ini memvalidasi penggunaan Mixed-Effects Model.

        #### Komponen Model:
        1.  **Fixed Effects (Efek Tetap):** Ini adalah variabel yang kita yakini memiliki pengaruh yang sama di semua provinsi. Dalam model ini, efek tetap meliputi:
            - Dana zakat di bidang: `Pendidikan`, `Kesehatan`, `Ekonomi`, `Kemanusiaan`.
            - `zakat_per_kapita` dan `total_zakat`.
            - `tahun` sebagai penanda waktu.
            Koefisien dari efek tetap ini menunjukkan rata-rata pengaruh variabel tersebut terhadap IPM di seluruh Indonesia.

        2.  **Random Effects (Efek Acak):** Ini adalah bagian yang paling menarik. Model ini memperbolehkan pengaruh variabel-variabel zakat untuk **berbeda di setiap provinsi**.
            - **Random Intercept:** Setiap provinsi memiliki "baseline" IPM yang berbeda, bahkan jika faktor zakatnya sama.
            - **Random Slope:** Efektivitas dana zakat (misalnya, dana pendidikan) bisa berbeda-beda. Di satu provinsi, kenaikan 1 miliar dana pendidikan bisa menaikkan IPM lebih tinggi dibandingkan provinsi lain.

        #### Teknik Regularisasi: Shrinkage
        Model ini cenderung *overfitting* pada data training. Untuk mengatasinya, teknik **Shrinkage** (penyusutan) diterapkan pada *random effects*. Koefisien random effect "ditarik" mendekati nol, sehingga mengurangi variasi ekstrem antar provinsi dan membuat model lebih generalisasi pada data baru. Tingkat shrinkage optimal (0.4) dipilih untuk menyeimbangkan antara performa model dan pencegahan overfitting.

        Dengan pendekatan ini, kita tidak hanya mendapatkan prediksi IPM, tetapi juga wawasan mengenai program zakat mana yang paling efektif di provinsi tertentu.
        """)

else:
    st.error("Gagal memuat file model atau data. Aplikasi tidak dapat berjalan.")
    st.info("Pastikan file `model_skema7H.pkl` dan `bps-od_15042_indeks_pmbngnn_manusia__prov_di_indonesia_data.xlsx` berada di direktori yang sama dengan `streamlit_app.py`.")