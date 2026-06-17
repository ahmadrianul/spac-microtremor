import streamlit as st
import numpy as np
import pandas as pd
import os
import shutil
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None
import matplotlib.colors as mcolors
from scipy.special import jv
from scipy.optimize import least_squares
import uuid

# Import spacunhas modules from the local folder
from spacunhas import environment, readdata, windowing, complexcoherence, spaccoefficient, dispersioncurve
import math

def suggest_similar_windows(pair_name, wins, path_processing, std_threshold=0.05, f_min_check=1.0, f_max_check=50.0):
    wins = [int(w) for w in wins]
    if len(wins) < 3:
        return sorted(wins), []
        
    f_interp = np.linspace(0, 50, 500)
    spac_matrix = []
    valid_wins = []
    
    for win in sorted(wins):
        fpath = os.path.join(path_processing, f"{pair_name}pair_spacfunc_window_{win}.txt")
        if os.path.exists(fpath):
            try:
                data = np.loadtxt(fpath, comments='#')
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                f = data[:, 0]
                c = data[:, 1]
                c_interp = np.interp(f_interp, f, c)
                spac_matrix.append(c_interp)
                valid_wins.append(win)
            except Exception:
                continue
                
    if not spac_matrix:
        return sorted(wins), []
        
    spac_matrix = np.array(spac_matrix)
    median_curve = np.median(spac_matrix, axis=0)
    idx_check = (f_interp >= f_min_check) & (f_interp <= f_max_check)
    
    errors = []
    for i in range(len(valid_wins)):
        diff = spac_matrix[i, idx_check] - median_curve[idx_check]
        rmse = np.sqrt(np.mean(diff**2))
        errors.append(rmse)
        
    errors = np.array(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    cutoff = mean_error + (std_threshold * std_error)
    
    is_similar = errors < cutoff
    suggested = [valid_wins[i] for i in range(len(valid_wins)) if is_similar[i]]
    outliers = [valid_wins[i] for i in range(len(valid_wins)) if not is_similar[i]]
    
    if not suggested:
        return sorted(wins), []
        
    return sorted(suggested), sorted(outliers)

def plot_coherence_grid_for_pair(pair_name, wins, path_processing):
    wins = [int(w) for w in wins]
    file_list = []
    for win in sorted(wins):
        fpath = os.path.join(path_processing, f"{pair_name}pair_spacfunc_window_{win}.txt")
        if os.path.exists(fpath):
            file_list.append((win, fpath))
            
    jumlah_file = len(file_list)
    if jumlah_file == 0:
        st.warning(f"Tidak ada file koherensi untuk {pair_name}")
        return
        
    n_cols = 3
    n_rows = math.ceil(jumlah_file / n_cols)
    
    fig_width = n_cols * 4.0
    fig_height = n_rows * 2.5
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        sharex=True, sharey=True
    )
    
    if not isinstance(axes, np.ndarray):
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
        
    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < jumlah_file:
            win_num, fpath = file_list[i]
            try:
                data = np.loadtxt(fpath, comments='#')
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                freq = data[:, 0]
                spac_val = data[:, 1]
                
                ax.plot(freq, spac_val, 'k-o', markersize=0.5, linewidth=0.5, alpha=0.7)
                ax.axhline(0, color='r', linestyle='--', linewidth=0.8, alpha=0.5)
                ax.set_title(f"Window {win_num}", fontsize=10, fontweight='bold')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_xlim(0, 50)
                ax.set_ylim(-1.1, 1.1)
            except Exception as e:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', color='red')
        else:
            ax.axis('off')
            
    fig.text(0.5, 0.005, 'Frekuensi (Hz)', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.005, 0.5, 'Koefisien SPAC', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    plt.suptitle(f"Grid Koherensi SPAC - {pair_name}", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    
    st.pyplot(fig)
    plt.close(fig)

# --- Page Configuration ---
st.set_page_config(
    page_title="SPAC Dispersion Extractor",
    page_icon="favicon.png",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    /* Force Light Background globally */
    .stApp {
        background-color: #FFFFFF !important;
        color: #111827 !important;
    }
    /* Style the Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
        border-right: 2px solid #E5E7EB !important;
    }
    .main-title {
        font-size: 2.5rem;
        color: #FF4B4B;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    /* Tab Styling with clear boundaries */
    div[data-baseweb="tab-list"] {
        border-bottom: 2px solid #E5E7EB !important;
        gap: 6px;
    }
    button[role="tab"] {
        border: 2px solid #E5E7EB !important;
        border-bottom: none !important;
        border-radius: 6px 6px 0 0 !important;
        background-color: #F3F4F6 !important;
        color: #4B5563 !important;
        font-weight: bold !important;
        padding: 8px 16px !important;
    }
    button[role="tab"][aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #FF4B4B !important;
        border: 2px solid #D1D5DB !important;
        border-bottom: 3px solid #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title Area
st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 0.5rem; margin-bottom: 1.5rem; text-align: center;">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 0.5rem;">
            <span style="font-family: 'Inter', 'Outfit', 'Montserrat', sans-serif; font-size: 56px; font-weight: 900; color: #111827; letter-spacing: -1.5px; line-height: 1;">SP</span>
            <svg width="50" height="50" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg" style="margin-left: 4px; margin-right: 4px; margin-bottom: 4px;">
                <!-- Triangle Lines representing letter A -->
                <line x1="6" y1="36" x2="22" y2="8" stroke="#FF4B4B" stroke-width="4.5" stroke-linecap="round" />
                <line x1="38" y1="36" x2="22" y2="8" stroke="#FF4B4B" stroke-width="4.5" stroke-linecap="round" />
                <line x1="6" y1="36" x2="38" y2="36" stroke="#FF4B4B" stroke-width="4.5" stroke-linecap="round" />
                <!-- Vertices & Center Donuts -->
                <!-- Top Vertex -->
                <circle cx="22" cy="8" r="5" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="2.5" />
                <!-- Bottom-Left Vertex -->
                <circle cx="6" cy="36" r="5" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="2.5" />
                <!-- Bottom-Right Vertex -->
                <circle cx="38" cy="36" r="5" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="2.5" />
                <!-- Center Node -->
                <circle cx="22" cy="26.7" r="5" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="2.5" />
            </svg>
            <span style="font-family: 'Inter', 'Outfit', 'Montserrat', sans-serif; font-size: 56px; font-weight: 900; color: #111827; letter-spacing: -1.5px; line-height: 1;">C</span>
        </div>
    </div>
""", unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="text-align: center; margin-top: -1rem;">Aplikasi Web Interaktif untuk Ekstraksi Kurva Dispersi Kecepatan Fase (Metode SPAC Sekuensial)</div>', unsafe_allow_html=True)

# --- Session State Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "step" not in st.session_state:
    st.session_state.step = "input"  # steps: input, computed_coherence, final_fit
if "coherence_calculated" not in st.session_state:
    st.session_state.coherence_calculated = False
if "data_list_by_pair" not in st.session_state:
    st.session_state.data_list_by_pair = {}
if "main_folder_path" not in st.session_state:
    st.session_state.main_folder_path = os.path.join("./workspace", st.session_state.session_id)

# --- Sidebar Parameter Inputs ---
st.sidebar.header("Parameter Pemrosesan")

working_dir = st.sidebar.text_input("Folder Direktori Kerja (Terisolasi per Sesi)", value=st.session_state.main_folder_path)
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
tab_teori, tab_input, tab_processing, tab_output, tab_inversion = st.tabs(["Teori & Panduan", "Input Data & Sesi", "Pemrosesan & Quality Control", "Kurva Dispersi Final", "Inversi Vs 1D"])

# ==========================================
# TAB 0: TEORI & PANDUAN
# ==========================================
with tab_teori:
    st.markdown("### Modul Teori & Panduan Metode SPAC Sekuensial")
    st.write("Silakan pilih topik penjelasan ilmiah di bawah ini untuk mempelajari lebih lanjut tentang metode SPAC, formulasi matematis, desain akuisisi, alur pengolahan, dan jurnal pendukung:")
    
    topik = st.selectbox(
        "Pilih Topik Penjelasan:",
        options=[
            "1. Pengantar & Konsep Dasar SPAC",
            "2. Formulasi Matematis & Rumus",
            "3. Desain Akuisisi Data Sekuensial (Two-Station)",
            "4. Alur Pengolahan Data (Step-by-Step)",
            "5. Dukungan Artikel & Jurnal Ilmiah"
        ],
        key="topik_teori"
    )
    
    st.markdown("---")
    
    if topik == "1. Pengantar & Konsep Dasar SPAC":
        st.markdown(r"""
            #### 1. Pengantar & Konsep Dasar SPAC (Spatial Autocorrelation)
            
            Metode Spatial Autocorrelation (SPAC) merupakan salah satu metode geofisika pasif non-invasif yang sangat populer digunakan untuk mengestimasi struktur kecepatan gelombang geser ($V_s$) satu dimensi (1D) bawah permukaan.
            
            ##### A. Prinsip Dasar
            Berbeda dengan metode seismik aktif tradisional (seperti MASW atau seismik refraksi) yang memerlukan sumber getaran buatan seperti palu godam (*sledgehammer*) atau peledak, metode SPAC memanfaatkan getaran alami bumi yang dikenal sebagai ambient noise atau mikrotremor. Getaran ini bersumber dari:
            *   **Aktivitas Alam:** Gelombang laut, pergerakan angin, interaksi atmosfer, aktivitas tektonik mikro.
            *   **Aktivitas Manusia:** Lalu lintas kendaraan, getaran mesin pabrik/industri, langkah kaki manusia, dan aktivitas perkotaan lainnya.
            
            Mikrotremor ini didominasi oleh gelombang permukaan (terutama gelombang Rayleigh) yang merambat secara horizontal di dekat permukaan bumi.
            
            ##### B. Keunggulan Metode SPAC
            1.  **Non-Invasif & Non-Destruktif:** Tidak memerlukan pengeboran lubang bor (seperti pada metode SPT atau downhole/crosshole) yang mahal dan dapat merusak permukaan tanah, sehingga sangat ramah lingkungan.
            2.  **Sangat Cocok untuk Kawasan Perkotaan (Urban Areas):** Di area padat penduduk, sangat sulit untuk membangkitkan getaran buatan berenergi besar (seperti dinamit atau palu). SPAC justru memanfaatkan kebisingan kota tersebut sebagai sumber sinyalnya.
            3.  **Investigasi Lebih Dalam:** Karena memanfaatkan gelombang permukaan frekuensi rendah dari ambient noise global, SPAC mampu mencapai kedalaman investigasi yang jauh lebih besar (dari puluhan hingga ratusan meter) dibandingkan metode aktif berenergi rendah.
            4.  **Ekonomis & Praktis:** Mengurangi biaya operasional lapangan dan mempermudah mobilisasi peralatan di area dengan topografi yang sulit.
            
            ##### C. Sejarah Singkat
            Teori Spatial Autocorrelation pertama kali diformulasikan secara matematis oleh geofisikawan Jepang, Keiiti Aki (1957). Beliau membuktikan secara matematis bahwa korelasi statistik dari getaran ambient noise yang direkam secara simultan pada beberapa stasiun penerima dapat digunakan untuk mengekstrak kurva dispersi gelombang permukaan. Teori ini kemudian dikembangkan dan disempurnakan menjadi metode praktis siap pakai untuk eksplorasi oleh Hiroshi Okada (2003) melalui bukunya yang terkenal, *"The Microtremor Survey Method"*.
        """)
        
    elif topik == "2. Formulasi Matematis & Rumus":
        st.markdown(r"""
            #### 2. Formulasi Matematis & Rumus-Rumus
            
            Berikut adalah formulasi matematis utama yang melandasi perambatan gelombang seismik, karakter dispersi gelombang Rayleigh, hingga perhitungan koefisien SPAC untuk mengekstrak kecepatan fase gelombang permukaan:
            
            ##### A. Kecepatan Gelombang Seismik Badan (Body Waves)
            Gelombang seismik badan merambat melalui interior bumi dan dibedakan menjadi Gelombang P (Primer/Longitudinal) dan Gelombang S (Sekunder/Transversal/Geser):
        """)
        
        st.latex(r"V_p = \sqrt{\frac{k + \frac{4}{3}\mu}{\rho}}")
        st.latex(r"V_s = \sqrt{\frac{\mu}{\rho}}")
        
        st.markdown(r"""
            Dimana:
            *   $V_p$ adalah kecepatan gelombang primer ($m/s$).
            *   $V_s$ adalah kecepatan gelombang geser ($m/s$).
            *   $k$ adalah Modulus Bulk medium ($N/m^2$).
            *   $\mu$ adalah Modulus Geser / Konstanta Lame kedua ($N/m^2$).
            *   $\rho$ adalah densitas massa medium ($kg/m^3$).
            
            ##### B. Kecepatan Gelombang Rayleigh ($V_R$)
            Gelombang Rayleigh menjalar di sepanjang permukaan bebas bumi dengan lintasan partikel elips retrograde. Pada medium elastis homogen dan isotropis dengan rasio Poisson $\nu = 0.25$, hubungan antara kecepatan gelombang Rayleigh ($V_R$) dan gelombang geser ($V_S$) didekati dengan:
        """)
        
        st.latex(r"V_R \approx 0.92 V_S")
        
        st.markdown(r"""
            ##### C. Karakter Dispersi Geometris Gelombang Rayleigh
            Gelombang Rayleigh bersifat dispersif pada medium berlapis, di mana komponen gelombang dengan frekuensi berbeda merambat dengan kecepatan fase yang berbeda karena kedalaman penetrasi yang berbeda. Kedalaman penetrasi efektif gelombang sebanding dengan panjang gelombangnya ($\lambda$). Secara matematis hubungan panjang gelombang, kecepatan fase ($c(f)$), dan frekuensi ($f$) dituliskan sebagai:
        """)
        
        st.latex(r"\lambda = \frac{c(f)}{f}")
        
        st.markdown(r"""
            *   **Frekuensi Tinggi (Panjang Gelombang Pendek):** Hanya merambat dan terkonsentrasi di lapisan dangkal yang lunak (kecepatan fase $c$ akan rendah).
            *   **Frekuensi Rendah (Panjang Gelombang Panjang):** Mampu menembus lebih dalam dan dipengaruhi oleh lapisan batuan yang lebih kaku (kecepatan fase $c$ akan tinggi).
            
            ##### D. Transformasi Fourier
            Untuk menganalisis frekuensi, data gelombang mentah dalam domain waktu-ruang $u(x, t)$ ditransformasikan ke domain frekuensi $U(x, \omega)$ melalui integral:
        """)
        
        st.latex(r"U(x, \omega) = \int_{-\infty}^{\infty} u(x, t) e^{-j\omega t} dt")
        
        st.markdown(r"""
            Dimana $\omega = 2\pi f$ adalah frekuensi sudut ($radian/detik$), $t$ adalah waktu ($detik$), dan $j$ adalah bilangan imajiner.
            
            ##### E. Fungsi Kompleks Koherensi ($C_{fg}(\omega)$)
            Koherensi dihitung antar pasangan receiver $f$ and $g$ yang dipisahkan oleh jarak tertentu:
        """)
        
        st.latex(r"C_{fg}(\omega) = \frac{CC_{fg}(\omega)}{|U_f(\omega)| \cdot |U_g(\omega)|}")
        
        st.markdown(r"""
            Di mana $CC_{fg}(\omega) = U_f(\omega) \cdot U_g^*(\omega)$ adalah korelasi silang (*cross-spectral density*) antara spektrum frekuensi tras pertama $U_f(\omega)$ dan konjugat kompleks spektrum tras kedua $U_g^*(\omega)$.
            
            ##### F. Koefisien SPAC Rata-rata Azimuthal ($\rho(f, r)$)
            Koefisien SPAC diperoleh dengan merata-ratakan nilai koherensi spasial $C(r, \theta, \omega)$ dari segala arah (azimuth $\theta$ dari $0$ hingga $2\pi$):
        """)
        
        st.latex(r"\rho(f, r) = \frac{1}{2\pi} \int_{0}^{2\pi} C(r, \theta, \omega) d\theta")
        
        st.markdown(r"""
            ##### G. Pencocokan ke Fungsi Bessel Jenis Pertama Orde Nol ($J_0$)
            Aki (1957) membuktikan bahwa jika medan gelombang mikrotremor datang secara acak dari segala arah dengan kekuatan seimbang, koefisien SPAC azimuthal rata-rata memiliki hubungan matematis langsung dengan Fungsi Bessel $J_0$:
        """)
        
        st.latex(r"\rho(f, r) = J_0(kr) = J_0\left(\frac{2\pi f r}{c(f)}\right)")
        
        st.markdown(r"""
            Dimana $r$ adalah jarak radius antar sensor ($meter$) dan $c(f)$ adalah kecepatan fase gelombang Rayleigh ($m/s$).
            
            ##### H. Estimasi Kecepatan Fase ($c(f)$)
            Dari nilai koefisien SPAC observasi lapangan $\rho(f, r)$ pada setiap frekuensi $f$, kita menyelesaikan persamaan transenden di atas untuk mencari argumen Bessel $x = kr$ (yaitu $x = \frac{2\pi f r}{c(f)}$) menggunakan metode numerik. Setelah nilai $x$ diperoleh, kecepatan fase gelombang Rayleigh dapat diekstrak dengan rumus:
        """)
        
        st.latex(r"c(f) = \frac{2\pi f r}{x}")
        
    elif topik == "3. Desain Akuisisi Data Sekuensial (Two-Station)":
        st.markdown(r"""
            #### 3. Desain Akuisisi Data Sekuensial (Two-Station Method)
            
            Secara konvensional, survei SPAC memerlukan susunan sensor (*array*) melingkar atau segitiga sama sisi yang merekam secara simultan (minimal menggunakan 4 geofon: 1 pusat, 3 keliling). Namun, konfigurasi tersebut membutuhkan banyak peralatan geofon dan kabel bentangan yang rumit.
            
            ##### A. Konsep Metode Dua Sensor (Two-Station Method)
            Untuk meningkatkan efisiensi di lapangan, metode SPAC Sekuensial (Two-Station Method) dapat diterapkan hanya dengan menggunakan 2 buah geofon (seperti unit sensor SmartSolo 3C).
            
            Sinyal direkam secara bertahap (sekuensial) dengan konfigurasi geometri segitiga sama sisi sebagai berikut:
            1.  **Geofon Referensi Tetap (Pusat A):** Diletakkan tetap di titik pusat koordinat selama seluruh sesi pengukuran.
            2.  **Geofon Bergerak (Rover B):** Dipindahkan secara berurutan ke stasiun keliling lingkaran pada radius $r$ yang sama, yaitu di titik B1, B2, dan B3 (membentuk sudut pisah $120^\circ$ satu sama lain).
            
            Perekaman dilakukan selama durasi tertentu pada masing-masing dari 3 stasiun pasangan (A-B1 pada Sesi 1, A-B2 pada Sesi 2, dan A-B3 pada Sesi 3).
        """)
        
        # Render larger SVG array geometry illustration
        st.markdown("""
            <div style="display: flex; justify-content: center; margin-top: 1rem; margin-bottom: 1rem; background-color: #FAFAFA; padding: 1.5rem; border: 2px solid #E5E7EB; border-radius: 8px;">
                <svg width="220" height="190" viewBox="0 0 200 170" xmlns="http://www.w3.org/2000/svg">
                    <!-- Triangle Lines -->
                    <line x1="100" y1="15" x2="35" y2="128" stroke="#FF4B4B" stroke-width="3.5" stroke-linecap="round" />
                    <line x1="100" y1="15" x2="165" y2="128" stroke="#FF4B4B" stroke-width="3.5" stroke-linecap="round" />
                    <line x1="35" y1="128" x2="165" y2="128" stroke="#FF4B4B" stroke-width="3.5" stroke-linecap="round" />
                    <!-- Radius lines (dashed) -->
                    <line x1="100" y1="90" x2="100" y2="15" stroke="#4B5563" stroke-width="1.5" stroke-dasharray="4,4" />
                    <line x1="100" y1="90" x2="35" y2="128" stroke="#4B5563" stroke-width="1.5" stroke-dasharray="4,4" />
                    <line x1="100" y1="90" x2="165" y2="128" stroke="#4B5563" stroke-width="1.5" stroke-dasharray="4,4" />
                    <!-- Center Node A -->
                    <circle cx="100" cy="90" r="9" fill="#FFFFFF" stroke="#111827" stroke-width="3" />
                    <text x="100" y="93.5" font-family="'Inter', sans-serif" font-size="10" font-weight="900" text-anchor="middle" fill="#111827">A</text>
                    <!-- Top Vertex B1 -->
                    <circle cx="100" cy="15" r="9" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="3" />
                    <text x="100" y="18.5" font-family="'Inter', sans-serif" font-size="10" font-weight="900" text-anchor="middle" fill="#FF4B4B">B1</text>
                    <!-- Bottom-Left Vertex B3 -->
                    <circle cx="35" cy="128" r="9" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="3" />
                    <text x="35" y="131.5" font-family="'Inter', sans-serif" font-size="10" font-weight="900" text-anchor="middle" fill="#FF4B4B">B3</text>
                    <!-- Bottom-Right Vertex B2 -->
                    <circle cx="165" cy="128" r="9" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="3" />
                    <text x="165" y="131.5" font-family="'Inter', sans-serif" font-size="10" font-weight="900" text-anchor="middle" fill="#FF4B4B">B2</text>
                    <!-- Radius Text -->
                    <text x="100" y="155" font-family="'Inter', sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#4B5563">Radius (r)</text>
                </svg>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(r"""
            ##### B. Parameter Akuisisi Data Lapangan
            *   **Radius Array ($r$):** $r$ (jari-jari array segitiga lapangan).
            *   **Durasi Perekaman:** minimal 20 menit untuk masing-masing stasiun pasangan (total ~60 menit perekaman).
            *   **Sampling Rate:** 100 Hz (diperoleh langsung dari metadata/header file rekaman seismik .sac/.mseed yang diunggah).
            *   **Sensor Geofon:** Geofon (komponen vertikal).
            
            ##### C. Asumsi Kunci
            Metode ini sangat bergantung pada asumsi bahwa medan getaran ambient noise bersifat stasioner (karakteristik statistiknya stabil dan tidak berubah) sepanjang total durasi perekaman ketiga sesi tersebut. Rata-rata azimuthal disimulasikan secara numerik saat pemrosesan data dengan merata-ratakan hasil koherensi silang dari ketiga stasiun pasangan yang direkam secara terpisah tersebut.
        """)
        
    elif topik == "4. Alur Pengolahan Data (Step-by-Step)":
        st.markdown(r"""
            #### 4. Alur Pengolahan Data (Step-by-Step)
            
            Proses pengolahan data mikrotremor metode SPAC sekuensial dari berkas mentah hingga menjadi kurva dispersi final dilakukan melalui tahapan berikut:
            
            1.  **Input Data Seismogram:**
                Membaca file rekaman biner seismik berformat SAC atau MINISEED untuk stasiun referensi (A) dan stasiun keliling (B) untuk masing-masing sesi (Pair 1, Pair 2, Pair 3).
            2.  **Segmentasi Waktu (Windowing):**
                Sinyal kontinu berdurasi panjang disegmentasi menjadi sejumlah jendela waktu yang lebih pendek (misalnya panjang jendela sebesar $2^{13} = 8192$ sampel atau setara 81.92 detik pada sampling rate 100 Hz) dengan *overlap* tertentu. Hal ini berguna untuk mengisolasi segmen data yang stasioner dan membuang gangguan transient noise lokal (seperti getaran akibat kendaraan yang lewat sangat dekat).
            3.  **Transformasi Spektral (FFT):**
                Setiap jendela waktu dari data domain waktu ditransformasikan ke domain frekuensi menggunakan algoritma Fast Fourier Transform (FFT) guna mendapatkan spektrum amplitudo dan fase sinyal.
            4.  **Kalkulasi Kompleks Koherensi:**
                Kompleks koherensi dihitung pada setiap frekuensi untuk setiap jendela waktu pada masing-masing sesi stasiun pasangan (A-B1, A-B2, A-B3).
            5.  **Quality Control (QC) & Seleksi Jendela Otomatis:**
                *   **Evaluasi RMSE:** Aplikasi menghitung selisih (Root Mean Square Error / RMSE) antara nilai koherensi setiap jendela terhadap nilai median koherensi pada rentang frekuensi kontrol (1 - 50 Hz).
                *   **Saran Window:** Jendela dengan RMSE di bawah batas cutoff (misalnya median + 0.05 standar deviasi) diklasifikasikan sebagai jendela bersih (konsisten), sedangkan jendela dengan RMSE tinggi diklasifikasikan sebagai outlier (noise) dan disarankan untuk dibuang.
            6.  **Rata-rata Dua Tahap (Two-Stage Stacking):**
                *   *Tahap 1:* Jendela-jendela yang terpilih di dalam masing-masing sesi dirata-ratakan secara independen untuk mendapatkan kurva koefisien SPAC Sesi 1 ($\rho_1$), Sesi 2 ($\rho_2$), dan Sesi 3 ($\rho_3$).
                *   *Tahap 2:* Ketiga kurva rata-rata sesi tersebut kemudian dirata-ratakan secara bersama-sama untuk mendapatkan kurva koefisien SPAC akhir ($\rho_{avg}$). Hal ini penting untuk menjamin pembobotan spasial yang seimbang dari segala arah sensor.
            7.  **Pencocokan Fungsi Bessel (Bessel Fitting):**
                Nilai koefisien SPAC rata-rata akhir $\rho_{avg}(f)$ dicocokkan dengan kurva Bessel teoritis $J_0(x)$ menggunakan metode kuadrat terkecil (*least-squares fitting*) to memperoleh nilai argumen Bessel $x = kr$ pada setiap frekuensi.
            8.  **Ekstraksi Kurva Dispersi:**
                Kecepatan fase Rayleigh dihitung menggunakan rumus $c(f) = \frac{2\pi f r}{x}$. Hasilnya kemudian disaring dalam batasan kecepatan fase minimum dan maksimum, serta dapat dipotong (*frequency cut*) pada rentang frekuensi yang bersih dari noise.
        """)
        
    elif topik == "5. Dukungan Artikel & Jurnal Ilmiah":
        st.markdown(r"""
            #### 5. Dukungan Artikel & Jurnal Ilmiah
            
            Berikut adalah daftar pustaka artikel ilmiah utama yang melandasi metode SPAC sekuensial:
            
            1.  Aki, K. (1957). Space and time spectra of stationary stochastic waves, with special reference to microtremors. *Bulletin of the Earthquake Research Institute*, 35, 415–456.
            2.  Chimoto, K., Onishi, K., & Matsuoka, T. (2023). Simple SPAC processing method using zero-crossing analysis of 1-bit data processing for real-time monitoring of velocity change. *Exploration Geophysics*, 1–13. https://doi.org/10.1186/s40623-023-01842-3
            3.  Cho, I. (2020). Two-sensor microtremor SPAC method: potential utility of imaginary spectrum components. *Geophysical Journal International*, 220(3), 1735–1747. https://doi.org/10.1093/gji/ggz454
            4.  Hayashi, K., Asten, M. W., Stephenson, W. J., Cornou, C., Hobiger, M., Pilz, M., & Yamanaka, H. (2022). Microtremor array method using spatial autocorrelation analysis of Rayleigh-wave data. *Journal of Seismology*, 26(4), 601–627. https://doi.org/10.1007/s10712-018-9474-2
            5.  Okada, H. (2003). *The Microtremor Survey Method* (K. Suto, Trans.). Society of Exploration Geophysicists.
        """)

# ==========================================
# TAB 1: INPUT DATA
# ==========================================
with tab_input:
    st.subheader("1. Unggah File Waveform (.sac / .mseed)")
    st.write("Unggah pasangan stasiun pusat (A) dan stasiun keliling (B) untuk masing-masing 3 sesi pemindahan alat:")

    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown("### Sesi 1 (A - B1)")
        st.markdown("<hr style='border: 0; height: 2px; background: linear-gradient(to right, #FF4B4B, rgba(255, 75, 75, 0.2)); margin-top: 4px; margin-bottom: 16px;' />", unsafe_allow_html=True)
        file_a1 = st.file_uploader("Upload Pusat A Sesi 1", type=["sac", "mseed"], key="file_a1")
        file_b1 = st.file_uploader("Upload Keliling B1 Sesi 1", type=["sac", "mseed"], key="file_b1")
        
    with col_s2:
        st.markdown("### Sesi 2 (A - B2)")
        st.markdown("<hr style='border: 0; height: 2px; background: linear-gradient(to right, #FF4B4B, rgba(255, 75, 75, 0.2)); margin-top: 4px; margin-bottom: 16px;' />", unsafe_allow_html=True)
        file_a2 = st.file_uploader("Upload Pusat A Sesi 2", type=["sac", "mseed"], key="file_a2")
        file_b2 = st.file_uploader("Upload Keliling B2 Sesi 2", type=["sac", "mseed"], key="file_b2")

    with col_s3:
        st.markdown("### Sesi 3 (A - B3)")
        st.markdown("<hr style='border: 0; height: 2px; background: linear-gradient(to right, #FF4B4B, rgba(255, 75, 75, 0.2)); margin-top: 4px; margin-bottom: 16px;' />", unsafe_allow_html=True)
        file_a3 = st.file_uploader("Upload Pusat A Sesi 3", type=["sac", "mseed"], key="file_a3")
        file_b3 = st.file_uploader("Upload Keliling B3 Sesi 3", type=["sac", "mseed"], key="file_b3")

    st.markdown("---")
    
    # Demo Data Option
    st.write("**Belum punya data lapangan?** Gunakan data demo mikrotremor kami untuk mencoba:")
    demo_btn = st.button("Gunakan Data Demo (Site Hasanuddin University)")
    
    if demo_btn:
        # We will copy demo data from local demo_data folder in the repo
        demo_src_dir = "./demo_data"
        if os.path.exists(demo_src_dir):
            try:
                for demo_file in ["A41.sac", "B41.sac", "A42.sac", "B42.sac", "A43.sac", "B43.sac"]:
                    src_path = os.path.join(demo_src_dir, demo_file)
                    dest_path = os.path.join(main_folder.path_data, demo_file)
                    shutil.copy(src_path, dest_path)
                st.success("Berhasil memuat Data Demo! Buka tab 'Pemrosesan & Quality Control' untuk mulai memproses.")
                st.session_state.demo_loaded = True
                st.session_state.data_list_by_pair = {
                    "Pair_1": ["A41.sac", "B41.sac"],
                    "Pair_2": ["A42.sac", "B42.sac"],
                    "Pair_3": ["A43.sac", "B43.sac"]
                }
            except Exception as ex:
                st.error(f"Gagal menyalin data demo: {ex}")
        else:
            st.warning("Folder data demo tidak ditemukan. Silakan unggah file Anda secara manual.")

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
        st.info(f"File siap untuk diproses: " + ", ".join([f"{k}: {v}" for k, v in st.session_state.data_list_by_pair.items()]))
    else:
        st.warning("Silakan unggah set data seismik minimal 1 sesi atau gunakan Data Demo.")

# ==========================================
# TAB 2: PROSES & QUALITY CONTROL
# ==========================================
with tab_processing:
    st.subheader("2. Pemrosesan Data & Stacking Window (Quality Control)")
    
    if len(st.session_state.data_list_by_pair) == 0:
        st.warning("Silakan unggah data Anda atau muat Data Demo di tab Input Data sebelum memproses.")
    else:
        # Run process button
        btn_run = st.button("Mulai Proses Windowing & Koherensi", type="primary")
        
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
                        st.markdown(f"#### Memproses Sesi {pair_key} ({files[0]} - {files[1]})")
                        
                        # Read data
                        data_init = readdata.ReadData(main_folder, files, 'SAC')
                        datas = data_init.read_data()
                        
                        # Show windowed data plots side-by-side
                        st.markdown(f"##### Visualisasi Windowed Data - Sesi {pair_key}")
                        col_w1, col_w2 = st.columns(2)
                        
                        # Windowing for each file
                        for idx_file, filename in enumerate(files):
                            # Use current config window size
                            data_windowing = windowing.Windowing(main_folder, data_init, window_size)
                            # Windowing returns self.n_window, self.x, self.y
                            list_data = data_windowing.windowing(filename)
                            
                            # Plot it and show in Streamlit
                            with (col_w1 if idx_file == 0 else col_w2):
                                st.write(f"Receiver: `{filename}`")
                                data_windowing.plot_windows()
                                fig = plt.gcf()
                                st.pyplot(fig)
                                plt.close(fig)
                            
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
            st.markdown("### Quality Control: Seleksi Jendela Koherensi")
            st.write("Visualisasikan semua jendela koherensi dan gunakan fitur saran otomatis untuk menyaring data yang mirip/konsisten:")

            # Initialize SPAC coefficient class using the first pair's info
            first_pair_key = list(st.session_state.data_list_by_pair.keys())[0]
            spaccoeff = spaccoefficient.SPACCoefficient(
                st.session_state.last_spac_coef_obj,
                st.session_state.windowing_objs[first_pair_key],
                radius=radius
            )
            
            # Fetch available files in processing
            list_coh, pairs_win, pairs = spaccoeff.list_coherence_file()

            # Group grid plots by pair (Outside the form so they can be viewed immediately without re-rendering the form)
            st.write("Klik expander di bawah ini untuk melihat kurva koherensi masing-masing jendela:")
            for pair_name, wins in sorted(pairs_win.items()):
                with st.expander(f"Tampilkan Grid Plot Koherensi - Sesi {pair_name}", expanded=False):
                    plot_coherence_grid_for_pair(pair_name, wins, os.path.join(working_dir, "processing"))

            st.write("Atur parameter filter dan lakukan seleksi jendela untuk stacking di bawah ini:")
            with st.form("qc_selection_form"):
                # Settings for Similarity Threshold
                st.write("**Parameter Saran Window Otomatis**")
                std_threshold = st.slider("Tingkat Ketatnya Seleksi (std multiplier)", min_value=0.01, max_value=0.50, value=0.05, step=0.01)
                f_min_check = st.number_input("Rentang Frekuensi Min (Hz)", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
                f_max_check = st.number_input("Rentang Frekuensi Max (Hz)", min_value=1.0, max_value=50.0, value=50.0, step=0.5)

                st.markdown("---")
                selected_pairs_win = {}
                
                # Group selections by pair
                for pair_name, wins in sorted(pairs_win.items()):
                    st.markdown(f"#### Analisis & Seleksi: Sesi {pair_name}")
                    
                    # 2. Get similarity suggestions
                    suggested_wins, outlier_wins = suggest_similar_windows(
                        pair_name, wins, os.path.join(working_dir, "processing"),
                        std_threshold=std_threshold, f_min_check=f_min_check, f_max_check=f_max_check
                    )
                    
                    # 3. Display suggestions
                    col_sug1, col_sug2 = st.columns(2)
                    with col_sug1:
                        st.success(f"**Saran Window Konsisten**: {suggested_wins}")
                    with col_sug2:
                        if outlier_wins:
                            st.warning(f"**Window Outlier (Tidak Disarankan)**: {outlier_wins}")
                        else:
                            st.info("**Window Outlier**: Tidak ada outlier terdeteksi.")
                    
                    # 4. Multiselect widget for this pair, default to suggested_wins
                    selected_wins = st.multiselect(
                        f"Pilih nomor window untuk di-stacking ({pair_name}):",
                        options=sorted([int(w) for w in wins]),
                        default=suggested_wins,
                        key=f"select_{pair_name}"
                    )
                    selected_pairs_win[pair_name] = selected_wins
                    st.markdown("---")
                
                # Submit button
                submit_qc = st.form_submit_button("Terapkan Seleksi & Hitung Stacking", type="primary")

            # --- Two-Stage Stacking ---
            pair_averages = {}
            freq = None

            for pair_name, sel_win in selected_pairs_win.items():
                if not sel_win:
                    continue
                all_rho = []
                for win in sel_win:
                    fpath = os.path.join(working_dir, "processing", f"{pair_name}pair_spacfunc_window_{win}.txt")
                    if os.path.exists(fpath):
                        data = np.loadtxt(fpath)
                        if data.ndim == 1:
                            data = data.reshape(1, -1)
                        freq = data[:, 0]
                        rho = data[:, 1]
                        all_rho.append(rho)
                if all_rho:
                    # Stage 1: Average for each pair
                    pair_averages[pair_name] = np.mean(all_rho, axis=0)

            if not pair_averages:
                st.error("Mohon pilih setidaknya 1 window bersih pada salah satu sesi untuk melakukan Stacking.")
            else:
                # Stage 2: Average across all active pairs
                avspac = np.mean(list(pair_averages.values()), axis=0)
                
                # Update spaccoeff properties so they are accessible to Tab 3
                spaccoeff.frequency = [freq]
                spaccoeff.spaccoefficient = [avspac]
                
                # Save stacked SPAC to file (as original code did)
                np.savetxt(f'{working_dir}/processing/avspac_radii{radius}.txt', 
                           np.column_stack((freq, avspac)), 
                           header='#Frequency (Hz) #SPACCoefficient')

                # Save objects for tab 3
                st.session_state.spaccoeff_obj = spaccoeff
                st.session_state.freq_stacked = freq
                st.session_state.avspac_stacked = avspac
                
                # --- Visualizations ---
                st.markdown("### Hasil Stacking SPAC Coefficient Akhir")
                st.write("Kurva di bawah ini adalah hasil rata-rata dua tahap (rata-rata per sesi, lalu dirata-ratakan antar sesi):")
                
                # Plotly Plot for observed SPAC coefficient
                fig_spac = go.Figure()
                fig_spac.add_trace(go.Scatter(
                    x=freq,
                    y=avspac,
                    mode='lines+markers',
                    name='Rata-rata Dua Tahap (Stacked)',
                    line=dict(color='black', width=2),
                    marker=dict(size=4)
                ))
                
                # Optional: Plot individual pair averages in distinct colors for comparison
                colors_list = ['#FF4B4B', '#3B82F6', '#10B981'] # Red, Blue, Green
                for idx_pair, (pair_name, pair_avg) in enumerate(sorted(pair_averages.items())):
                    color = colors_list[idx_pair % len(colors_list)]
                    fig_spac.add_trace(go.Scatter(
                        x=freq,
                        y=pair_avg,
                        mode='lines',
                        name=f'Rata-rata Sesi {pair_name}',
                        line=dict(color=color, width=1.5, dash='dash')
                    ))
                
                fig_spac.update_layout(
                     title=dict(
                         text="Koefisien SPAC Akhir vs Frekuensi",
                         font=dict(color="black")
                     ),
                     xaxis=dict(
                         title=dict(text="Frekuensi (Hz)", font=dict(color="black")),
                         range=[0, 50],
                         showgrid=True,
                         gridcolor="lightgray",
                         linecolor="black",
                         ticks="outside",
                         tickcolor="black",
                         tickfont=dict(color="black")
                     ),
                     yaxis=dict(
                         title=dict(text="Koefisien SPAC", font=dict(color="black")),
                         range=[-1.1, 1.1],
                         showgrid=True,
                         gridcolor="lightgray",
                         linecolor="black",
                         ticks="outside",
                         tickcolor="black",
                         tickfont=dict(color="black")
                     ),
                     template="plotly_white",
                     height=500,
                     plot_bgcolor="white",
                     paper_bgcolor="white"
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
        st.warning("Silakan selesaikan pemrosesan dan Quality Control di tab QC Data terlebih dahulu.")
    else:
        st.info(f"Menghitung kurva dispersi dengan batas kecepatan fase: {min_pv} m/s hingga {max_pv} m/s.")
        
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
                
                # --- Frequency Cut Feature ---
                st.markdown("### Pemotongan Rentang Frekuensi Kurva Dispersi")
                st.write("Sesuaikan rentang frekuensi yang ingin Anda gunakan untuk analisis lebih lanjut:")
                col_cut1, col_cut2 = st.columns(2)
                with col_cut1:
                    f_min_val = float(np.round(np.min(f_pv), 2))
                    cut_f_min = st.number_input("Frekuensi Minimum Cut (Hz)", min_value=0.0, max_value=100.0, value=max(0.0, f_min_val), step=0.1)
                with col_cut2:
                    f_max_val = float(np.round(np.max(f_pv), 2))
                    cut_f_max = st.number_input("Frekuensi Maximum Cut (Hz)", min_value=0.0, max_value=100.0, value=min(50.0, f_max_val), step=0.1)
                
                # Apply filter
                mask_cut = (f_pv >= cut_f_min) & (f_pv <= cut_f_max)
                f_pv_cut = f_pv[mask_cut]
                pv_cut = pv[mask_cut]
                x_bessel_cut = x_bessel[mask_cut]
                bessel_fit_cut = bessel_fit[mask_cut]
                avspac_dec_cut = avspac_dec[mask_cut]

                # Save to session state for Inversion tab
                st.session_state.f_pv_cut = f_pv_cut
                st.session_state.pv_cut = pv_cut
                st.session_state.active_radius = radius

                # --- Visualizations ---
                col_c1, col_c2 = st.columns(2)
                
                with col_c1:
                    st.markdown("#### Fitting Fungsi Bessel J0")
                    
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
                    
                    # Observed Data points plotted at the calculated Bessel coordinates (Show all points in gray, active in black)
                    fig_bessel.add_trace(go.Scatter(
                        x=x_bessel,
                        y=avspac_dec,
                        mode='markers',
                        name='Data Lapangan (Luar Rentang)',
                        marker=dict(color='lightgray', size=5),
                        showlegend=True
                    ))
                    
                    fig_bessel.add_trace(go.Scatter(
                        x=x_bessel_cut,
                        y=avspac_dec_cut,
                        mode='markers',
                        name='Data Lapangan (Aktif)',
                        marker=dict(color='black', size=7)
                    ))
                    
                    fig_bessel.update_layout(
                        title=dict(
                            text="Pencocokan Data Lapangan ke Fungsi Bessel J0",
                            font=dict(color="black")
                        ),
                        xaxis=dict(
                            title=dict(text="Bessel Argument (2*pi*f*r / c)", font=dict(color="black")),
                            range=[0, 10],
                            showgrid=True,
                            gridcolor="lightgray",
                            linecolor="black",
                            ticks="outside",
                            tickcolor="black",
                            tickfont=dict(color="black")
                        ),
                        yaxis=dict(
                            title=dict(text="Koefisien SPAC", font=dict(color="black")),
                            range=[-1.1, 1.1],
                            showgrid=True,
                            gridcolor="lightgray",
                            linecolor="black",
                            ticks="outside",
                            tickcolor="black",
                            tickfont=dict(color="black")
                        ),
                        template="plotly_white",
                        height=450,
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        legend=dict(font=dict(color="black"))
                    )
                    fig_bessel.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_bessel, use_container_width=True)
                    
                with col_c2:
                    st.markdown("#### Kurva Dispersi Rayleigh")
                    
                    # Tentukan batas zoom otomatis secara dinamis
                    if len(f_pv_cut) > 0:
                        y_min = float(np.min(pv_cut))
                        y_max = float(np.max(pv_cut))
                        y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 50.0
                        y_range = [y_min - y_pad, y_max + y_pad]
                        x_range = [float(cut_f_min), float(cut_f_max)]
                    else:
                        x_range = [0.0, 50.0]
                        y_range = [float(min_pv), float(max_pv)]
                        
                    fig_disp = go.Figure()
                    
                    # Full Curve (Gray)
                    fig_disp.add_trace(go.Scatter(
                        x=f_pv,
                        y=pv,
                        mode='lines+markers',
                        name='Semua Frekuensi',
                        line=dict(color='lightgray', width=1),
                        marker=dict(color='lightgray', size=4)
                    ))
                    
                    # Cut Curve (Blue)
                    fig_disp.add_trace(go.Scatter(
                        x=f_pv_cut,
                        y=pv_cut,
                        mode='lines+markers',
                        name='Rentang Terpotong (Aktif)',
                        line=dict(color='blue', width=2.5),
                        marker=dict(color='darkblue', size=6)
                    ))
                    
                    fig_disp.update_layout(
                         title=dict(
                             text="Kurva Dispersi Rayleigh (Kecepatan Fase vs Frekuensi)",
                             font=dict(color="black")
                         ),
                         xaxis=dict(
                             title=dict(text="Frekuensi (Hz)", font=dict(color="black")),
                             range=x_range,
                             showgrid=True,
                             gridcolor="lightgray",
                             linecolor="black",
                             ticks="outside",
                             tickcolor="black",
                             tickfont=dict(color="black")
                         ),
                         yaxis=dict(
                             title=dict(text="Kecepatan Fase (m/s)", font=dict(color="black")),
                             range=y_range,
                             showgrid=True,
                             gridcolor="lightgray",
                             linecolor="black",
                             ticks="outside",
                             tickcolor="black",
                             tickfont=dict(color="black")
                         ),
                         template="plotly_white",
                         height=450,
                         plot_bgcolor="white",
                         paper_bgcolor="white",
                         legend=dict(font=dict(color="black"))
                     )
                    fig_disp.update_yaxes(type="linear")
                    st.plotly_chart(fig_disp, use_container_width=True)
                    
                # --- Export and Download Results ---
                st.markdown("---")
                st.markdown("### Ekspor Hasil Analisis")
                
                # Full Dataframe
                export_df_full = pd.DataFrame({
                    "Frequency_Hz": f_pv,
                    "Phase_Velocity_ms": pv,
                    "Bessel_Argument": x_bessel,
                    "Bessel_Fit": bessel_fit,
                    "SPAC_Coefficient_Observed": avspac_dec
                })
                
                # Cut Dataframe
                export_df_cut = pd.DataFrame({
                    "Frequency_Hz": f_pv_cut,
                    "Phase_Velocity_ms": pv_cut,
                    "Bessel_Argument": x_bessel_cut,
                    "Bessel_Fit": bessel_fit_cut,
                    "SPAC_Coefficient_Observed": avspac_dec_cut
                })
                
                # Show preview of cut data
                st.write("Pratinjau Data Kurva Dispersi Terpotong (Aktif):")
                st.dataframe(export_df_cut, use_container_width=True)
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    csv_full = export_df_full.to_csv(index=False)
                    st.download_button(
                        label="Unduh Kurva Dispersi Penuh (CSV)",
                        data=csv_full,
                        file_name=f"dispersion_curve_FULL_radius_{radius}m.csv",
                        mime="text/csv",
                        key="btn_full"
                    )
                with col_btn2:
                    csv_cut = export_df_cut.to_csv(index=False)
                    st.download_button(
                        label="Unduh Kurva Dispersi Terpotong (CSV)",
                        data=csv_cut,
                        file_name=f"dispersion_curve_CUT_{cut_f_min:.1f}to{cut_f_max:.1f}Hz.csv",
                        mime="text/csv",
                        key="btn_cut"
                    )
                

                
            except Exception as ex_fit:
                st.error(f"Gagal melakukan least-squares fitting: {ex_fit}")

# ==========================================
# TAB 5: INVERSI VS 1D
# ==========================================
with tab_inversion:
    st.subheader("4. Inversi Linier Terbobot (1D Vs Profile Inversion)")
    
    if "f_pv_cut" not in st.session_state or len(st.session_state.f_pv_cut) == 0:
        st.warning("Silakan selesaikan pemotongan rentang frekuensi kurva dispersi di tab 'Kurva Dispersi Final' terlebih dahulu.")
    else:
        # Load active data
        f_obs = st.session_state.f_pv_cut
        c_obs = st.session_state.pv_cut
        active_r = st.session_state.get("active_radius", 4.62)
        
        # Calculate wavelengths
        wavelengths = c_obs / f_obs
        max_wl = float(np.max(wavelengths))
        min_wl = float(np.min(wavelengths))
        
        # Suggested Parameters
        suggested_max_depth = max_wl * 0.5
        suggested_min_depth = min_wl * 0.4
        
        st.markdown("### Konfigurasi Parameter Inversi")
        st.write("Parameter di bawah ini diisi otomatis sebagai saran berdasarkan karakteristik panjang gelombang data Anda:")
        
        # Layout columns for parameters
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown("#### Parameter Konversi Empiris")
            coeff_depth = st.number_input("Koefisien Kedalaman (a)", min_value=0.1, max_value=1.0, value=0.4, step=0.05, 
                                          help="Kedalaman diestimasi dengan z = a * lambda. Default 0.4")
            coeff_vel = st.number_input("Koefisien Kecepatan Vs (b)", min_value=0.5, max_value=2.0, value=1.1, step=0.05,
                                        help="Vs diestimasi dengan Vs = b * c. Default 1.1")
            
            st.markdown("#### Parameter Bobot Sinyal (Inversi)")
            m_weight = st.number_input("Eksponen Pembobotan Kedalaman (m)", min_value=0.5, max_value=5.0, value=1.5, step=0.1,
                                       help="Mempengaruhi seberapa cepat kepekaan Rayleigh wave memudar terhadap kedalaman. Default 1.5")
            alpha_damping = st.number_input("Faktor Redaman / Damping (alpha)", min_value=0.0, max_value=10.0, value=0.1, step=0.01,
                                            help="Mengontrol kemulusan perlapisan Vs. Semakin besar nilai, profil semakin mulus/seragam.")
            
        with col_p2:
            st.markdown("#### Konfigurasi Perlapisan (Layering)")
            num_layers = st.slider("Jumlah Lapisan (N)", min_value=2, max_value=8, value=4, step=1)
            
            st.markdown("**Batas Kedalaman Bawah Lapisan (m):**")
            # Automatically suggest depths based on suggested_max_depth
            depth_boundaries = [0.0]
            depths_ok = True
            for i in range(1, num_layers):
                # Suggested boundary spacing: linear intervals up to suggested_max_depth
                suggested_val = float(np.round(suggested_max_depth * (i / num_layers), 1))
                val = st.number_input(f"Batas Bawah Lapisan {i} (meter)", min_value=0.1, value=suggested_val, step=0.5, key=f"d_bound_{i}")
                if i > 1 and val <= depth_boundaries[-1]:
                    st.error(f"Error: Batas bawah Lapisan {i} harus lebih besar dari Lapisan {i-1} ({depth_boundaries[-1]} m).")
                    depths_ok = False
                depth_boundaries.append(val)
            
        if depths_ok:
            # Prepare depths array
            d_bounds = np.array(depth_boundaries)
            
            # Inversion computation
            M_data = len(f_obs)
            W_mat = np.zeros((M_data, num_layers))
            for i in range(M_data):
                wl = wavelengths[i]
                for k in range(num_layers):
                    d_start = d_bounds[k]
                    d_end = d_bounds[k+1] if k < num_layers-1 else np.inf
                    W_mat[i, k] = np.exp(-m_weight * d_start / wl) - (np.exp(-m_weight * d_end / wl) if k < num_layers-1 else 0.0)
            
            # Residual function
            def residuals_inv(Vs_params, W, c_data, alpha):
                res_fit = np.dot(W, Vs_params) - c_data
                res_smooth = np.zeros(len(Vs_params) - 1)
                for i in range(len(Vs_params) - 1):
                    res_smooth[i] = np.sqrt(alpha) * (Vs_params[i+1] - Vs_params[i])
                return np.concatenate([res_fit, res_smooth])
                
            # Initial guess (average of empirical Vs)
            emp_vs_vals = coeff_vel * c_obs
            emp_z_vals = coeff_depth * wavelengths
            vs0_guess = np.ones(num_layers) * np.mean(emp_vs_vals)
            
            with st.spinner("Sedang melakukan inversi least-squares..."):
                try:
                    # Inversion using bounded least squares
                    res_opt = least_squares(residuals_inv, vs0_guess, args=(W_mat, c_obs, alpha_damping), bounds=(50.0, 2000.0))
                    Vs_inverted = res_opt.x
                    c_calc = np.dot(W_mat, Vs_inverted)
                    
                    st.success("Inversi berhasil diselesaikan!")
                    
                    # --- Plotting Results ---
                    col_plot1, col_plot2 = st.columns(2)
                    
                    with col_plot1:
                        st.markdown("#### 1D Shear-Wave Velocity (Vs) Profile")
                        
                        # Construct step function for Vs profile
                        plot_z = []
                        plot_vs = []
                        max_plot_depth = float(np.max(emp_z_vals)) * 1.2 if len(emp_z_vals) > 0 else 50.0
                        
                        for k in range(num_layers):
                            z_top = d_bounds[k]
                            z_bottom = d_bounds[k+1] if k < num_layers-1 else max_plot_depth
                            plot_z.extend([z_top, z_bottom])
                            plot_vs.extend([Vs_inverted[k], Vs_inverted[k]])
                            
                        fig_vs = go.Figure()
                        
                        # Inverted Layered Profile (Step Line)
                        fig_vs.add_trace(go.Scatter(
                            x=plot_vs,
                            y=plot_z,
                            mode='lines',
                            name='Vs Terinversi (Layered)',
                            line=dict(color='red', width=3)
                        ))
                        
                        # Empirical scatter points
                        fig_vs.add_trace(go.Scatter(
                            x=emp_vs_vals,
                            y=emp_z_vals,
                            mode='markers',
                            name='Estimasi Empiris (Scatter)',
                            marker=dict(color='blue', size=6, symbol='circle')
                        ))
                        
                        fig_vs.update_layout(
                            title=dict(text="Profil Vs 1D terhadap Kedalaman", font=dict(color="black")),
                            xaxis=dict(
                                title=dict(text="Kecepatan Gelombang Geser Vs (m/s)", font=dict(color="black")),
                                showgrid=True, gridcolor="lightgray", linecolor="black", ticks="outside", tickcolor="black", tickfont=dict(color="black")
                            ),
                            yaxis=dict(
                                title=dict(text="Kedalaman z (m)", font=dict(color="black")),
                                autorange="reverse",  # Depth points downwards
                                showgrid=True, gridcolor="lightgray", linecolor="black", ticks="outside", tickcolor="black", tickfont=dict(color="black")
                            ),
                            template="plotly_white",
                            height=450,
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            legend=dict(font=dict(color="black"))
                        )
                        st.plotly_chart(fig_vs, use_container_width=True)
                        
                    with col_plot2:
                        st.markdown("#### Pencocokan Kurva Dispersi (Goodness of Fit)")
                        
                        fig_fit = go.Figure()
                        
                        # Observed
                        fig_fit.add_trace(go.Scatter(
                            x=f_obs,
                            y=c_obs,
                            mode='markers+lines',
                            name='Observed (Hasil Fitting SPAC)',
                            line=dict(color='black', width=1),
                            marker=dict(color='black', size=6)
                        ))
                        
                        # Calculated
                        fig_fit.add_trace(go.Scatter(
                            x=f_obs,
                            y=c_calc,
                            mode='lines',
                            name='Calculated (Dari Model Vs)',
                            line=dict(color='blue', width=2.5, dash='dash')
                        ))
                        
                        # Dynamic limits
                        y_min_fit = float(min(np.min(c_obs), np.min(c_calc)))
                        y_max_fit = float(max(np.max(c_obs), np.max(c_calc)))
                        y_pad_fit = (y_max_fit - y_min_fit) * 0.05 if y_max_fit > y_min_fit else 50.0
                        
                        fig_fit.update_layout(
                            title=dict(text="Perbandingan Kurva Dispersi Observasi vs Kalkulasi", font=dict(color="black")),
                            xaxis=dict(
                                title=dict(text="Frekuensi (Hz)", font=dict(color="black")),
                                showgrid=True, gridcolor="lightgray", linecolor="black", ticks="outside", tickcolor="black", tickfont=dict(color="black")
                            ),
                            yaxis=dict(
                                title=dict(text="Kecepatan Fase (m/s)", font=dict(color="black")),
                                range=[y_min_fit - y_pad_fit, y_max_fit + y_pad_fit],
                                showgrid=True, gridcolor="lightgray", linecolor="black", ticks="outside", tickcolor="black", tickfont=dict(color="black")
                            ),
                            template="plotly_white",
                            height=450,
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            legend=dict(font=dict(color="black"))
                        )
                        st.plotly_chart(fig_fit, use_container_width=True)
                        
                    # --- Table and Download Inversion Results ---
                    st.markdown("---")
                    st.markdown("### Hasil Perlapisan Vs Terinversi")
                    
                    # Construct results table
                    layers_list = []
                    for k in range(num_layers):
                        z_top = d_bounds[k]
                        z_bottom = d_bounds[k+1] if k < num_layers-1 else np.inf
                        thick = z_bottom - z_top if k < num_layers-1 else np.nan
                        vs_val = Vs_inverted[k]
                        
                        # Get soil class label
                        if vs_val > 1500:
                            s_class = "SA (Batuan Keras)"
                        elif vs_val > 750:
                            s_class = "SB (Batuan)"
                        elif vs_val > 350:
                            s_class = "SC (Tanah Keras / Batuan Lunak)"
                        elif vs_val > 175:
                            s_class = "SD (Tanah Sedang)"
                        else:
                            s_class = "SE (Tanah Lunak)"
                            
                        layers_list.append({
                            "Lapisan": k + 1,
                            "Kedalaman Atas (m)": z_top,
                            "Kedalaman Bawah (m)": z_bottom if k < num_layers-1 else "Infinity",
                            "Ketebalan (m)": thick if k < num_layers-1 else "Infinity",
                            "Kecepatan Vs (m/s)": np.round(vs_val, 1),
                            "Kelas Situs Tanah (FEMA/SNI)": s_class
                        })
                        
                    df_res = pd.DataFrame(layers_list)
                    st.dataframe(df_res, use_container_width=True)
                    
                    # CSV Download
                    csv_res = df_res.to_csv(index=False)
                    st.download_button(
                        label="Unduh Profil Vs Terinversi (CSV)",
                        data=csv_res,
                        file_name=f"vs_profile_inversion_layers_{num_layers}.csv",
                        mime="text/csv",
                        key="btn_download_vs"
                    )
                    
                except Exception as ex_inv:
                    st.error(f"Gagal melakukan proses inversi: {ex_inv}")
