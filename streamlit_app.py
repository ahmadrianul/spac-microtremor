import streamlit as st
import numpy as np
import pandas as pd
import os
import shutil
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import jv

# Import spacunhas modules from the local folder
from spacunhas import environment, readdata, windowing, complexcoherence, spaccoefficient, dispersioncurve

# --- Page Configuration ---
st.set_page_config(
    page_title="SPAC Dispersion Extractor",
    page_icon="🌋",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #FF4B4B;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6D7A96;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #E9ECEF;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title Area
st.markdown('<div class="main-title">🌋 SPAC Microtremor Dispersion Extractor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Aplikasi Web Interaktif untuk Ekstraksi Kurva Dispersi Kecepatan Fase (Metode SPAC Sekuensial)</div>', unsafe_allow_html=True)

# --- Session State Initialization ---
if "step" not in st.session_state:
    st.session_state.step = "input"  # steps: input, computed_coherence, final_fit
if "coherence_calculated" not in st.session_state:
    st.session_state.coherence_calculated = False
if "data_list_by_pair" not in st.session_state:
    st.session_state.data_list_by_pair = {}
if "main_folder_path" not in st.session_state:
    st.session_state.main_folder_path = "./workspace"

# --- Sidebar Parameter Inputs ---
st.sidebar.header("⚙️ Parameter Pemrosesan")

working_dir = st.sidebar.text_input("Folder Direktori Kerja", value=st.session_state.main_folder_path)
st.session_state.main_folder_path = working_dir

radius = st.sidebar.number_input("Jarak Radius Array (m)", min_value=0.1, value=4.62, step=0.01)
fs = st.sidebar.number_input("Sampling Rate (Hz)", min_value=1.0, value=100.0, step=1.0)
smooth_constant = st.sidebar.number_input("Konstanta Smoothing Spektral", min_value=5, value=100, step=5)

# Selection of window size
window_power = st.sidebar.slider("Ukuran Jendela (2^N)", min_value=10, max_value=16, value=13)
window_size = 2**window_power
st.sidebar.caption(f"Panjang Jendela: {window_size} sampel ({window_size/fs:.2f} detik)")

decimation_factor = st.sidebar.number_input("Faktor Desimasi Fitting", min_value=1, value=10, step=1)
min_pv = st.sidebar.number_input("Batas Min Kecepatan Fase (m/s)", min_value=10.0, value=100.0, step=10.0)
max_pv = st.sidebar.number_input("Batas Max Kecepatan Fase (m/s)", min_value=50.0, value=1000.0, step=10.0)

# --- Initialize Environment ---
@st.cache_resource
def get_environment(path):
    return environment.SPACProcessing(path)

try:
    main_folder = get_environment(working_dir)
except Exception as e:
    st.error(f"Gagal menginisialisasi folder kerja: {e}")
    st.stop()

# --- Main Layout Tabs ---
tab_input, tab_processing, tab_output = st.tabs(["📂 Input Data & Sesi", "🔄 Pemrosesan & Quality Control", "📊 Kurva Dispersi Final"])

# ==========================================
# TAB 1: INPUT DATA
# ==========================================
with tab_input:
    st.subheader("1. Unggah File Waveform (.sac / .mseed)")
    st.write("Unggah pasangan stasiun pusat (A) dan stasiun keliling (B) untuk masing-masing 3 sesi pemindahan alat:")

    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🔹 Sesi 1 (A - B1)")
        file_a1 = st.file_uploader("Upload Pusat A Sesi 1", type=["sac", "mseed"], key="file_a1")
        file_b1 = st.file_uploader("Upload Keliling B1 Sesi 1", type=["sac", "mseed"], key="file_b1")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_s2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🔹 Sesi 2 (A - B2)")
        file_a2 = st.file_uploader("Upload Pusat A Sesi 2", type=["sac", "mseed"], key="file_a2")
        file_b2 = st.file_uploader("Upload Keliling B2 Sesi 2", type=["sac", "mseed"], key="file_b2")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_s3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🔹 Sesi 3 (A - B3)")
        file_a3 = st.file_uploader("Upload Pusat A Sesi 3", type=["sac", "mseed"], key="file_a3")
        file_b3 = st.file_uploader("Upload Keliling B3 Sesi 3", type=["sac", "mseed"], key="file_b3")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Demo Data Option
    st.write("💡 **Belum punya data lapangan?** Gunakan data demo mikrotremor kami untuk mencoba:")
    demo_btn = st.button("Gunakan Data Demo (Site Hasanuddin University)")
    
    if demo_btn:
        # We will copy demo data from local C:\Users\RIANO\spac-unhas-main\spac-unhas-main\examples\data2\data
        demo_src_dir = r"C:\Users\RIANO\spac-unhas-main\spac-unhas-main\examples\data2\data"
        if os.path.exists(demo_src_dir):
            try:
                for demo_file in ["A11.sac", "B11.sac", "A12.sac", "B12.sac", "A13.sac", "B13.sac"]:
                    src_path = os.path.join(demo_src_dir, demo_file)
                    dest_path = os.path.join(main_folder.path_data, demo_file)
                    shutil.copy(src_path, dest_path)
                st.success("Berhasil memuat Data Demo! Buka tab 'Pemrosesan & Quality Control' untuk mulai memproses.")
                st.session_state.demo_loaded = True
                st.session_state.data_list_by_pair = {
                    "Pair_1": ["A11.sac", "B11.sac"],
                    "Pair_2": ["A12.sac", "B12.sac"],
                    "Pair_3": ["A13.sac", "B13.sac"]
                }
            except Exception as ex:
                st.error(f"Gagal menyalin data demo: {ex}")
        else:
            st.warning("Folder data demo lokal tidak ditemukan. Silakan unggah file Anda secara manual.")

    # Save Uploaded Files
    if not demo_btn:
        pairs_to_process = {}
        # Sesi 1
        if file_a1 and file_b1:
            with open(os.path.join(main_folder.path_data, file_a1.name), "wb") as f:
                f.write(file_a1.getbuffer())
            with open(os.path.join(main_folder.path_data, file_b1.name), "wb") as f:
                f.write(file_b1.getbuffer())
            pairs_to_process["Pair_1"] = [file_a1.name, file_b1.name]
            
        # Sesi 2
        if file_a2 and file_b2:
            with open(os.path.join(main_folder.path_data, file_a2.name), "wb") as f:
                f.write(file_a2.getbuffer())
            with open(os.path.join(main_folder.path_data, file_b2.name), "wb") as f:
                f.write(file_b2.getbuffer())
            pairs_to_process["Pair_2"] = [file_a2.name, file_b2.name]

        # Sesi 3
        if file_a3 and file_b3:
            with open(os.path.join(main_folder.path_data, file_a3.name), "wb") as f:
                f.write(file_a3.getbuffer())
            with open(os.path.join(main_folder.path_data, file_b3.name), "wb") as f:
                f.write(file_b3.getbuffer())
            pairs_to_process["Pair_3"] = [file_a3.name, file_b3.name]

        if len(pairs_to_process) > 0:
            st.session_state.data_list_by_pair = pairs_to_process

    # Display status
    if len(st.session_state.data_list_by_pair) > 0:
        st.info(f"📂 File siap untuk diproses: " + ", ".join([f"{k}: {v}" for k, v in st.session_state.data_list_by_pair.items()]))
    else:
        st.warning("Silakan unggah set data seismik minimal 1 sesi atau gunakan Data Demo.")

# ==========================================
# TAB 2: PROSES & QUALITY CONTROL
# ==========================================
with tab_processing:
    st.subheader("2. Pemrosesan Data & Stacking Window (Quality Control)")
    
    if len(st.session_state.data_list_by_pair) == 0:
        st.warning("Silakan unggah data Anda atau muat Data Demo di tab pertama sebelum memproses.")
    else:
        # Run process button
        btn_run = st.button("🚀 Mulai Proses Windowing & Koherensi", type="primary")
        
        if btn_run or st.session_state.coherence_calculated:
            with st.spinner("Sedang memproses sinyal, membagi jendela, dan menghitung koherensi..."):
                try:
                    # Clean processing files first on fresh run
                    if btn_run:
                        for item in os.listdir(main_folder.path_processing):
                            if item.endswith(".txt"):
                                os.remove(os.path.join(main_folder.path_processing, item))
                        for item in os.listdir(main_folder.path_figures):
                            if item.endswith(".png"):
                                os.remove(os.path.join(main_folder.path_figures, item))

                    # Object to hold all computed windowing objects
                    windowing_objs = {}
                    last_spac_coef_obj = None
                    n_windows_found = {}

                    # Loop through each pair
                    for pair_key, files in st.session_state.data_list_by_pair.items():
                        st.markdown(f"#### 🔍 Memproses Sesi {pair_key} ({files[0]} - {files[1]})")
                        
                        # Read data
                        data_init = readdata.ReadData(main_folder, files, 'SAC')
                        datas = data_init.read_data()
                        
                        # Windowing for each file
                        for filename in files:
                            # Use current config window size
                            data_windowing = windowing.Windowing(main_folder, data_init, window_size)
                            # Windowing returns self.n_window, self.x, self.y
                            list_data = data_windowing.windowing(filename)
                            
                        # Save window info
                        windowing_objs[pair_key] = data_windowing
                        n_windows = list_data[0]
                        n_windows_found[pair_key] = n_windows
                        
                        # Calculate complex coherence for each window
                        for win_idx in range(1, n_windows + 1):
                            spac_coefficient = complexcoherence.ComplexCoherence(
                                main_folder, data_windowing, fs=fs, smooth_constant=smooth_constant
                            )
                            # calculate coherence
                            f, coh = spac_coefficient.calculate_coherence(files[0], files[1], window_no=win_idx)
                            # Plotting using internal Matplotlib method (creates png)
                            spac_coefficient.plot_spacfunc(50)
                            last_spac_coef_obj = spac_coefficient

                    st.session_state.coherence_calculated = True
                    st.success("Pemrosesan Waveform selesai! Silakan lakukan Quality Control di bawah ini.")
                    
                    # Store computed values in session state for dynamic UI update
                    st.session_state.windowing_objs = windowing_objs
                    st.session_state.last_spac_coef_obj = last_spac_coef_obj
                    st.session_state.n_windows_found = n_windows_found

                except Exception as ex:
                    st.error(f"Terjadi kesalahan saat memproses data: {ex}")
                    st.session_state.coherence_calculated = False

        # --- Quality Control Section (Dynamic UI) ---
        if st.session_state.coherence_calculated:
            st.markdown("---")
            st.markdown("### 🛠️ Quality Control: Seleksi Jendela Koherensi")
            st.write("Peta koherensi per jendela waktu telah dihitung. Pilih jendela waktu yang bersih (bebas dari transisi noise atau spike) untuk dirata-ratakan (stacking):")

            # Determine maximum windows among the computed pairs
            max_win = max(st.session_state.n_windows_found.values())
            all_wins = list(range(1, max_win + 1))
            
            # Interactive Multi-select widget
            # Default selections (e.g. typical clean windows from user's thesis)
            default_clean = [w for w in all_wins if w not in [1, 4, 7]] if len(all_wins) >= 7 else all_wins
            window_bersih = st.multiselect(
                "Pilih nomor window bersih yang akan di-stacking:",
                options=all_wins,
                default=default_clean
            )
            
            if not window_bersih:
                st.warning("Mohon pilih setidaknya 1 window bersih.")
            else:
                # Initialize SPAC coefficient class
                # Note: SPACCoefficient constructor takes (coherence_processing, windowing_processing, radius)
                # We use the last coherence and windowing object to initialize
                first_pair_key = list(st.session_state.data_list_by_pair.keys())[0]
                spaccoeff = spaccoefficient.SPACCoefficient(
                    st.session_state.last_spac_coef_obj,
                    st.session_state.windowing_objs[first_pair_key],
                    radius=radius
                )
                
                # Fetch available files in processing
                list_coh, pairs_win, pairs = spaccoeff.list_coherence_file()
                
                # Setup selected_pairs_win dictionary
                selected_pairs_win = {}
                for pair_name in pairs_win.keys():
                    selected_pairs_win[pair_name] = window_bersih
                
                # Calculate average SPAC
                freq, avspac = spaccoeff.avspac((list_coh, pairs_win, pairs), selected_pairs_win)
                
                # Save objects for tab 3
                st.session_state.spaccoeff_obj = spaccoeff
                st.session_state.freq_stacked = freq
                st.session_state.avspac_stacked = avspac
                
                # --- Visualizations ---
                st.markdown("#### 📈 Hasil Stacking SPAC Coefficient")
                
                # Plotly Plot for observed SPAC coefficient
                fig_spac = go.Figure()
                fig_spac.add_trace(go.Scatter(
                    x=freq,
                    y=avspac,
                    mode='lines+markers',
                    name='Rata-rata (Stacked)',
                    line=dict(color='black', width=2),
                    marker=dict(size=4)
                ))
                
                # Optional: Plot individual window coherences in light gray
                # We can load files and plot them
                try:
                    for pair_name, wins in pairs_win.items():
                        for win in wins:
                            file_p = os.path.join(working_dir, "processing", f"{pair_name}pair_spacfunc_window_{win}.txt")
                            if os.path.exists(file_p):
                                single_data = np.loadtxt(file_p)
                                fig_spac.add_trace(go.Scatter(
                                    x=single_data[:, 0],
                                    y=single_data[:, 1],
                                    mode='lines',
                                    line=dict(color='lightgray', width=0.5),
                                    showlegend=False
                                ))
                except Exception:
                    pass
                
                fig_spac.update_layout(
                    title="Koefisien SPAC Rata-rata vs Frekuensi",
                    xaxis_title="Frekuensi (Hz)",
                    yaxis_title="Koefisien SPAC",
                    xaxis_range=[0, 50],
                    yaxis_range=[-1.1, 1.1],
                    template="plotly_white",
                    height=500
                )
                fig_spac.add_hline(y=0, line_dash="dash", line_color="red", line_width=1)
                
                st.plotly_chart(fig_spac, use_container_width=True)
                
                st.success("Koefisien SPAC rata-rata berhasil ditumpuk! Buka tab 'Kurva Dispersi Final' untuk melihat hasil pencocokan.")

# ==========================================
# TAB 3: DISPERSION CURVE FINAL
# ==========================================
with tab_output:
    st.subheader("3. Kurva Dispersi Final (Pencocokan Fungsi Bessel J0)")
    
    if "spaccoeff_obj" not in st.session_state:
        st.warning("Silakan selesaikan pemrosesan dan Quality Control di tab kedua terlebih dahulu.")
    else:
        st.info(f"📊 Menghitung kurva dispersi dengan batas kecepatan fase: {min_pv} m/s hingga {max_pv} m/s.")
        
        # Inisialisasi Objek DispersionCurve
        dc = dispersioncurve.DispersionCurve(st.session_state.spaccoeff_obj)
        
        # Decimator
        # decimator reduces samples count to speed up the least-square fitting
        freq_dec, avspac_dec = dc.decimator(decimation_factor)
        
        # Calculate dispersion curve
        with st.spinner("Sedang mengepaskan koefisien SPAC ke Fungsi Bessel (Least-Squares Fitting)..."):
            try:
                # returns: pv, f_pv, bessel_fit, x_bessel, fitted_avspac
                pv, f_pv, bessel_fit, x_bessel, fitted_avspac = dc.calculate_dispcurv(min_pv, max_pv)
                
                # --- Visualizations ---
                col_c1, col_c2 = st.columns(2)
                
                with col_c1:
                    st.markdown("#### 🎯 Fitting Fungsi Bessel J0")
                    # Plot observed SPAC coefficients vs Bessel fit
                    # Bessel arg: x = (2 * pi * f * r) / pv
                    # Observed y: avspac_dec
                    # Fitted y: J_0(x)
                    
                    fig_bessel = go.Figure()
                    
                    # Ideal J0 Curve
                    x_ideal = np.linspace(0, 15, 200)
                    y_ideal = jv(0, x_ideal)
                    fig_bessel.add_trace(go.Scatter(
                        x=x_ideal,
                        y=y_ideal,
                        mode='lines',
                        name='Fungsi Bessel J0 Ideal',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Observed Data points plotted at the calculated Bessel coordinates
                    fig_bessel.add_trace(go.Scatter(
                        x=x_bessel,
                        y=avspac_dec,
                        mode='markers',
                        name='Data Lapangan (Observed)',
                        marker=dict(color='black', size=6)
                    ))
                    
                    fig_bessel.update_layout(
                        title="Pencocokan Data Lapangan ke Fungsi Bessel J0",
                        xaxis_title="Bessel Argument (2*pi*f*r / c)",
                        yaxis_title="Koefisien SPAC",
                        xaxis_range=[0, 10],
                        yaxis_range=[-1.1, 1.1],
                        template="plotly_white",
                        height=450
                    )
                    fig_bessel.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_bessel, use_container_width=True)
                    
                with col_c2:
                    st.markdown("#### 📈 Kurva Dispersi Rayleigh")
                    
                    # Plotly scatter for Dispersion Curve
                    fig_disp = go.Figure()
                    fig_disp.add_trace(go.Scatter(
                        x=f_pv,
                        y=pv,
                        mode='lines+markers',
                        name='Kurva Dispersi',
                        line=dict(color='blue', width=2),
                        marker=dict(color='darkblue', size=5)
                    ))
                    
                    fig_disp.update_layout(
                        title="Kurva Dispersi Rayleigh (Kecepatan Fase vs Frekuensi)",
                        xaxis_title="Frekuensi (Hz)",
                        yaxis_title="Kecepatan Fase (m/s)",
                        xaxis_range=[0, 50],
                        yaxis_range=[min_pv, max_pv],
                        template="plotly_white",
                        height=450
                    )
                    fig_disp.update_yaxes(type="linear")
                    st.plotly_chart(fig_disp, use_container_width=True)
                    
                # --- Export and Download Results ---
                st.markdown("---")
                st.markdown("### 📥 Ekspor Hasil Analisis")
                
                # Save dispersion data to a pandas dataframe
                export_df = pd.DataFrame({
                    "Frequency_Hz": f_pv,
                    "Phase_Velocity_ms": pv,
                    "Bessel_Argument": x_bessel,
                    "Bessel_Fit": bessel_fit,
                    "SPAC_Coefficient_Observed": avspac_dec
                })
                
                # Show preview
                st.write("Pratinjau Data Kurva Dispersi:")
                st.dataframe(export_df.head(10), use_container_width=True)
                
                # Download button
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="💾 Unduh Kurva Dispersi (CSV)",
                    data=csv_data,
                    file_name=f"dispersion_curve_radius_{radius}m.csv",
                    mime="text/csv"
                )
                

                
            except Exception as ex_fit:
                st.error(f"Gagal melakukan least-squares fitting: {ex_fit}")
