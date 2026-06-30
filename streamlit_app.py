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


# =============================================================================
# MONTE CARLO FORWARD ENGINE FUNCTIONS
# =============================================================================
def _rs_t_vec(c, vp, vs):
    vp = np.maximum(vp, 1.0)
    vs = np.maximum(vs, 1.0)
    r = np.sqrt(1.0 - (c / vp) ** 2 + 0j)
    s = np.sqrt(1.0 - (c / vs) ** 2 + 0j)
    t = 2.0 - (c / vs) ** 2
    return r, s, t

def _CS_vec(x, Cm=80.0):
    C = np.cosh(x)
    S = np.sinh(x)
    real_x = np.real(x)
    mask = np.abs(real_x) > Cm
    if np.any(mask):
        fac = np.exp(real_x[mask])
        C[mask] /= fac
        S[mask] /= fac
    return C, S

def D_fast_delta_vec(c_arr, k_arr, h, vp, vs, rho):
    n = len(h)
    size = len(c_arr)
    x1 = np.zeros(size, dtype=complex); x2 = np.zeros(size, dtype=complex)
    x3 = np.zeros(size, dtype=complex); x4 = np.zeros(size, dtype=complex)
    x5 = np.zeros(size, dtype=complex); x6 = np.ones(size, dtype=complex)

    for i in range(n):
        r, s, t = _rs_t_vec(c_arr, vp[i], vs[i])
        arg_p = k_arr * r * h[i]
        arg_s = k_arr * s * h[i]
        Cp, Sp = _CS_vec(arg_p)
        Cs, Ss = _CS_vec(arg_s)

        p1 = Cp*x1 + Ss*x4; p2 = Cp*x2 + Ss*x5
        p3 = -Sp*x1 + Cp*x3; p4 = -Sp*x2 + Cp*x6
        q2 = Cs*p2 - (r/s)*Ss*p4; q3 = Cs*p3 - (r/s)*Ss*p1; q4 = -Ss*p2 + Cs*p4

        mu_i = rho[i]*(vs[i]**2)
        mu_ip1 = rho[i+1]*(vs[i+1]**2)
        y = mu_ip1/mu_i; q = (vs[i+1]/vs[i])**2
        a = 1.0-q; ap = 1.0-1.0/q; b = 1.0-y; bp = 1.0-1.0/y

        x1 = a*q2 + b*p1; x2 = a*q3 + b*p2; x3 = ap*q4 + bp*p3
        x4 = a*p1 + b*q2; x5 = a*p2 + b*q3; x6 = ap*p3 + bp*q4

    rL, sL, tL = _rs_t_vec(c_arr, vp[-1], vs[-1])
    D = 2.0*sL*x2 - (rL*(tL**2)*sL)*x1
    return D

def calc_curve_fast_optimized(freq_hz, h, vp, vs, rho, c_obs):
    c_out = np.full_like(freq_hz, np.nan, dtype=float)
    c_min_scan = max(100.0, np.min(c_obs) * 0.6)
    c_max_scan = np.max(c_obs) * 1.6
    c_scan = np.linspace(c_min_scan, c_max_scan, 500) 
    
    for ii, f in enumerate(freq_hz):
        k_scan = 2.0 * np.pi * f / c_scan
        D_vals = np.real(D_fast_delta_vec(c_scan, k_scan, h, vp, vs, rho))
        signs = np.sign(D_vals)
        crossings = np.where(signs[:-1] * signs[1:] < 0)[0]
        
        if len(crossings) > 0:
            idx = crossings[0]
            c1, c2 = c_scan[idx], c_scan[idx+1]
            val_1 = np.real(D_fast_delta_vec(np.array([c1]), np.array([2.0*np.pi*f/c1]), h, vp, vs, rho))[0]
            for _ in range(10): 
                cm = 0.5 * (c1 + c2)
                val_m = np.real(D_fast_delta_vec(np.array([cm]), np.array([2.0*np.pi*f/cm]), h, vp, vs, rho))[0]
                if np.sign(val_1) * np.sign(val_m) <= 0:
                    c2 = cm
                else:
                    c1 = cm
                    val_1 = val_m
            c_out[ii] = 0.5 * (c1 + c2)
    return c_out

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

def load_uploaded_file_as_stream(uploaded_file, fs):
    import io
    import obspy
    import numpy as np
    from obspy import Trace, Stream
    from obspy.core import UTCDateTime
    
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()
    
    is_ascii = any(filename.endswith(ext) for ext in [".txt", ".asc", ".dat", ".csv"])
    
    if is_ascii:
        try:
            content = file_bytes.decode("utf-8")
            lines = content.strip().split("\n")
            data_values = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith(";"):
                    continue
                parts = line.replace(",", " ").split()
                if not parts:
                    continue
                try:
                    if len(parts) >= 2:
                        val = float(parts[1])
                    else:
                        val = float(parts[0])
                    data_values.append(val)
                except ValueError:
                    continue
            
            if len(data_values) > 0:
                data_array = np.array(data_values, dtype=np.float32)
                tr = Trace(data=data_array)
                tr.stats.sampling_rate = fs
                tr.stats.station = 'ASCII'
                tr.stats.channel = 'Z'
                tr.stats.network = 'XX'
                tr.stats.starttime = UTCDateTime(0)
                return Stream(traces=[tr])
        except Exception:
            pass

    # Guess format from extension
    fmt = None
    if filename.endswith(".sac"):
        fmt = "SAC"
    elif any(filename.endswith(ext) for ext in [".mseed", ".miniseed", ".msd"]):
        fmt = "MSEED"

    try:
        mem_file = io.BytesIO(file_bytes)
        stream = obspy.read(mem_file, format=fmt)
        return stream
    except Exception as ex:
        # Try alternate format
        try:
            mem_file = io.BytesIO(file_bytes)
            other_fmt = "MSEED" if fmt == "SAC" else "SAC"
            stream = obspy.read(mem_file, format=other_fmt)
            return stream
        except Exception:
            # Try autodetect
            try:
                mem_file = io.BytesIO(file_bytes)
                stream = obspy.read(mem_file)
                return stream
            except Exception:
                pass
                
        # Fallback to ASCII parsing
        try:
            content = file_bytes.decode("utf-8")
            lines = content.strip().split("\n")
            data_values = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith(";"):
                    continue
                parts = line.replace(",", " ").split()
                if not parts:
                    continue
                try:
                    if len(parts) >= 2:
                        val = float(parts[1])
                    else:
                        val = float(parts[0])
                    data_values.append(val)
                except ValueError:
                    continue
            if len(data_values) > 0:
                data_array = np.array(data_values, dtype=np.float32)
                tr = Trace(data=data_array)
                tr.stats.sampling_rate = fs
                tr.stats.station = 'ASCII'
                tr.stats.channel = 'Z'
                tr.stats.network = 'XX'
                tr.stats.starttime = UTCDateTime(0)
                return Stream(traces=[tr])
        except Exception:
            pass
        st.error(f"Gagal membaca file `{uploaded_file.name}`. Format tidak dikenali atau file rusak.")
        return None

def plot_trace_preview(trace, title):
    import numpy as np
    import matplotlib.pyplot as plt
    
    data = trace.data
    npts = len(data)
    fs = trace.stats.sampling_rate
    times = np.arange(npts) / fs
    
    # Decimate data for preview to make it render instantly
    max_pts = 1000
    if npts > max_pts:
        step = npts // max_pts
        preview_data = data[::step]
        preview_times = times[::step]
    else:
        preview_data = data
        preview_times = times
        
    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.plot(preview_times, preview_data, color="#FF4B4B", linewidth=0.6, alpha=0.9)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=4)
    ax.set_xlabel("Waktu (s)", fontsize=7, labelpad=2)
    ax.set_ylabel("Amplitudo", fontsize=7, labelpad=2)
    ax.tick_params(axis='both', which='major', labelsize=7, pad=1)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
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
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 0.5rem; gap: 12px;">
            <span style="font-family: 'Inter', 'Outfit', 'Montserrat', sans-serif; font-size: 56px; font-weight: 900; color: #111827; line-height: 1;">S</span>
            <span style="font-family: 'Inter', 'Outfit', 'Montserrat', sans-serif; font-size: 56px; font-weight: 900; color: #111827; line-height: 1;">P</span>
            <svg width="50" height="50" viewBox="0 0 50 50" xmlns="http://www.w3.org/2000/svg" style="display: block; transform: translateY(2px);">
                <!-- Circular Track for array -->
                <circle cx="25" cy="29.2" r="18.6" fill="none" stroke="#FF4B4B" stroke-width="2.2" stroke-dasharray="3,3" />
                <!-- Triangle Lines representing letter A -->
                <line x1="9" y1="38.5" x2="25" y2="10.5" stroke="#FF4B4B" stroke-width="4.5" stroke-linecap="round" />
                <line x1="41" y1="38.5" x2="25" y2="10.5" stroke="#FF4B4B" stroke-width="4.5" stroke-linecap="round" />
                <line x1="9" y1="38.5" x2="41" y2="38.5" stroke="#FF4B4B" stroke-width="4.5" stroke-linecap="round" />
                <!-- Vertices & Center Donuts -->
                <!-- Top Vertex -->
                <circle cx="25" cy="10.5" r="5" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="2.5" />
                <!-- Bottom-Left Vertex -->
                <circle cx="9" cy="38.5" r="5" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="2.5" />
                <!-- Bottom-Right Vertex -->
                <circle cx="41" cy="38.5" r="5" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="2.5" />
                <!-- Center Node -->
                <circle cx="25" cy="29.2" r="5" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="2.5" />
            </svg>
            <span style="font-family: 'Inter', 'Outfit', 'Montserrat', sans-serif; font-size: 56px; font-weight: 900; color: #111827; line-height: 1;">C</span>
        </div>
    </div>
""", unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="text-align: center; margin-top: -1rem;">Interactive Web Application for Phase Velocity Dispersion Curve Extraction (Sequential SPAC Method)</div>', unsafe_allow_html=True)

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
st.sidebar.header("Processing Parameters")

working_dir = st.sidebar.text_input("Working Directory Folder (Session-Isolated)", value=st.session_state.main_folder_path)
st.session_state.main_folder_path = working_dir

radius = st.sidebar.number_input("Array Radius (m)", min_value=0.1, value=4.62, step=0.01)
fs = st.sidebar.number_input("Sampling Rate (Hz)", min_value=1.0, value=100.0, step=1.0)
smooth_constant = st.sidebar.number_input("Spectral Smoothing Constant", min_value=5, value=100, step=5)

# Selection of window size
window_power = st.sidebar.slider("Window Size (2^N)", min_value=10, max_value=16, value=13)
window_size = 2**window_power
st.sidebar.caption(f"Window Length: {window_size} samples ({window_size/fs:.2f} seconds)")

decimation_factor = st.sidebar.number_input("Fitting Decimation Factor", min_value=1, value=10, step=1)
min_pv = st.sidebar.number_input("Min Phase Velocity Limit (m/s)", min_value=10.0, value=100.0, step=10.0)
max_pv = st.sidebar.number_input("Max Phase Velocity Limit (m/s)", min_value=50.0, value=1000.0, step=10.0)

# --- Initialize Environment ---
@st.cache_resource
def get_environment(path):
    return environment.SPACProcessing(path)

try:
    main_folder = get_environment(working_dir)
except Exception as e:
    st.error(f"Failed to initialize working directory: {e}")
    st.stop()

# --- Main Layout Tabs ---
tab_teori, tab_input, tab_processing, tab_output = st.tabs(["Theory & Guidelines", "Data Input & Sessions", "Processing & Quality Control", "Dispersion Curve"])
# ==========================================
# TAB 0: TEORI & PANDUAN
# ==========================================
with tab_teori:
    st.markdown("### Theoretical Module & Guidelines for the Sequential SPAC Method")
    st.write("Select a scientific topic below to learn more about the Spatial Autocorrelation (SPAC) method, its mathematical formulation, acquisition design, processing workflow, and supporting literature:")
    
    topik = st.selectbox(
        "Select Explanation Topic:",
        options=[
            "1. Introduction & Fundamental Concepts of SPAC",
            "2. Mathematical Formulation & Equations",
            "3. Sequential Data Acquisition Design (Two-Station)",
            "4. Data Processing Workflow (Step-by-Step)",
            "5. Supporting Literature & Scientific Journals",
            "6. Acknowledgments"
        ],
        key="topik_teori"
    )
    
    st.markdown("---")
    
    if topik == "1. Introduction & Fundamental Concepts of SPAC":
        st.markdown(r"""
            #### 1. Introduction & Fundamental Concepts of SPAC (Spatial Autocorrelation)
            
            The Spatial Autocorrelation (SPAC) method is a popular passive, non-invasive geophysical method used to estimate the one-dimensional (1D) shear wave velocity ($V_s$) structure of the subsurface.
            
            ##### A. Fundamental Principles
            Unlike traditional active seismic methods (such as MASW or seismic refraction) which require artificial seismic sources like a sledgehammer or explosives, the SPAC method utilizes natural earth vibrations known as ambient noise or microtremors. These vibrations originate from:
            *   **Natural Activities:** Ocean waves, wind interactions, atmospheric variations, and microtectonic activities.
            *   **Human Activities:** Traffic, factory/industrial machinery, human footsteps, and other urban activities.
            
            These microtremors are dominated by surface waves (mainly Rayleigh waves) that propagate horizontally near the Earth's surface.
            
            ##### B. Key Advantages of the SPAC Method
            1.  **Non-Invasive & Non-Destructive:** Does not require borehole drilling (unlike SPT or downhole/crosshole methods), making it cost-effective and environmentally friendly.
            2.  **Highly Suitable for Urban Areas:** In densely populated areas, generating high-energy artificial vibrations (such as dynamites or hammers) is challenging. SPAC turns urban noise into a useful signal source.
            3.  **Deeper Investigation Depth:** By leveraging low-frequency surface waves from global ambient noise, SPAC can achieve significantly greater depths of investigation (ranging from tens to hundreds of meters) compared to active low-energy source methods.
            4.  **Cost-Effective & Practical:** Minimizes field operational costs and simplifies equipment logistics in challenging terrains.
            
            ##### C. Brief History
            The theory of Spatial Autocorrelation was first mathematically formulated by Japanese geophysicist Keiiti Aki (1957). He proved that the statistical correlation of ambient noise recorded simultaneously at multiple receiving stations can be used to extract the dispersion curve of surface waves. This theory was later developed and refined into a practical exploration tool by Hiroshi Okada (2003) in his renowned book, *"The Microtremor Survey Method"*.
        """)
        
    elif topik == "2. Mathematical Formulation & Equations":
        st.markdown(r"""
            #### 2. Mathematical Formulation & Equations
            
            Below are the primary mathematical formulations governing seismic wave propagation, the geometric dispersion character of Rayleigh waves, and the calculation of the SPAC coefficient to extract surface wave phase velocity:
            
            ##### A. Body Wave Velocities
            Body waves propagate through the Earth's interior and are classified into P-waves (Primary/Longitudinal) and S-waves (Secondary/Transverse/Shear):
        """)
        
        st.latex(r"V_p = \sqrt{\frac{k + \frac{4}{3}\mu}{\rho}}")
        st.latex(r"V_s = \sqrt{\frac{\mu}{\rho}}")
        
        st.markdown(r"""
            Where:
            *   $V_p$ is the primary wave velocity ($m/s$).
            *   $V_s$ is the shear wave velocity ($m/s$).
            *   $k$ is the bulk modulus of the medium ($N/m^2$).
            *   $\mu$ is the shear modulus / second Lame constant ($N/m^2$).
            *   $\rho$ is the mass density of the medium ($kg/m^3$).
            
            ##### B. Geometric Dispersion Character of Rayleigh Waves
            Rayleigh waves are dispersive in a layered medium, meaning that wave components with different frequencies propagate at different phase velocities due to their varying penetration depths. The effective penetration depth of a wave is proportional to its wavelength ($\lambda$). Mathematically, the relation between wavelength, phase velocity ($c(f)$), and frequency ($f$) is written as:
        """)
        
        st.latex(r"\lambda = \frac{c(f)}{f}")
        
        st.markdown(r"""
            *   **High Frequencies (Short Wavelengths):** Only propagate and concentrate within shallow, soft layers (resulting in low phase velocity $c$).
            *   **Low Frequencies (Long Wavelengths):** Penetrate deeper and are influenced by stiffer, deeper rock layers (resulting in high phase velocity $c$).
            
            ##### C. Fourier Transform
            To analyze frequency components, raw wave data in the space-time domain $u(x, t)$ is transformed into the frequency domain $U(x, \omega)$ via the following integral:
        """)
        
        st.latex(r"U(x, \omega) = \int_{-\infty}^{\infty} u(x, t) e^{-j\omega t} dt")
        
        st.markdown(r"""
            Where $\omega = 2\pi f$ is the angular frequency ($rad/s$), $t$ is time ($s$), and $j$ is the imaginary unit.
            
            ##### D. Complex Coherence Function ($C_{fg}(\omega)$)
            Coherence is computed between receiver pairs $f$ and $g$ separated by a distance:
        """)
        
        st.latex(r"C_{fg}(\omega) = \frac{CC_{fg}(\omega)}{|U_f(\omega)| \cdot |U_g(\omega)|}")
        
        st.markdown(r"""
            Where $CC_{fg}(\omega) = U_f(\omega) \cdot U_g^*(\omega)$ is the cross-spectral density between the spectrum of the first trace $U_f(\omega)$ and the complex conjugate spectrum of the second trace $U_g^*(\omega)$.
            
            ##### E. Azimuthally Averaged SPAC Coefficient ($\rho(f, r)$)
            The SPAC coefficient is obtained by averaging the spatial coherence $C(r, \theta, \omega)$ over all directions (azimuth $\theta$ from $0$ to $2\pi$):
        """)
        
        st.latex(r"\rho(f, r) = \frac{1}{2\pi} \int_{0}^{2\pi} C(r, \theta, \omega) d\theta")
        
        st.markdown(r"""
            ##### F. Fitting to Zero-Order Bessel Function of the First Kind ($J_0$)
            Aki (1957) demonstrated that if the ambient noise field arrives randomly from all directions with equal power, the azimuthally averaged SPAC coefficient has a direct mathematical relationship with the Bessel function $J_0$:
        """)
        
        st.latex(r"\rho(f, r) = J_0(kr) = J_0\left(\frac{2\pi f r}{c(f)}\right)")
        
        st.markdown(r"""
            Where $r$ is the array radius ($m$) and $c(f)$ is the Rayleigh wave phase velocity ($m/s$).
            
            ##### G. Phase Velocity Estimation ($c(f)$)
            From the observed SPAC coefficient $\rho(f, r)$ at each frequency $f$, we solve the transcendental equation above to find the Bessel argument $x = kr$ (i.e., $x = \frac{2\pi f r}{c(f)}$) using numerical methods. Once $x$ is determined, the Rayleigh wave phase velocity can be extracted using the formula:
        """)
        
        st.latex(r"c(f) = \frac{2\pi f r}{x}")
        
    elif topik == "3. Sequential Data Acquisition Design (Two-Station)":
        st.markdown(r"""
            #### 3. Sequential Data Acquisition Design (Two-Station Method)
            
            Conventionally, SPAC surveys require a circular sensor array recording simultaneously (e.g., 1 center sensor and 3 or more perimeter sensors). However, this configuration demands multiple geophones and complex cabling in the field.
            
            ##### A. Concept of the Two-Sensor Method
            To improve field efficiency, the Sequential SPAC method (Two-Station Method) is applied using only 2 geophones (such as SmartSolo 3C wireless geophone units).
            
            Signals are recorded sequentially (in stages) with the following geophone geometry configuration:
            1.  **Fixed Reference Geophone (Center A):** Placed permanently at the center of the circle during all recording sessions.
            2.  **Mobile Geophone (Rover B):** Moved sequentially to perimeter stations on the same radius $r$ at various points ($B_1, B_2, \dots, B_n$).
            
            This method can be implemented in two major array geometries:
            *   **Equilateral Triangle Geometry (3 Sessions):** The mobile geophone is moved to 3 perimeter stations ($B_1, B_2, B_3$) separated by $120^\circ$ angles (requires 3 recording sessions).
            *   **General Circular Geometry (N Sessions):** The mobile geophone is moved to $N$ circular perimeter stations ($B_1, B_2, \dots, B_n$) to achieve higher azimuthal resolution (requires $N$ recording sessions).
        """)
        
        # Render both SVG diagrams side-by-side
        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            st.markdown("<div style='text-align: center; font-weight: bold; margin-bottom: 0.5rem; color: #111827;'>A. Equilateral Triangle Geometry (3 Sessions)</div>", unsafe_allow_html=True)
            st.markdown("""<div style="display: flex; justify-content: center; background-color: #FAFAFA; padding: 1.5rem; border: 2px solid #E5E7EB; border-radius: 8px;">
<svg width="200" height="170" viewBox="0 0 200 170" xmlns="http://www.w3.org/2000/svg">
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
</div>""", unsafe_allow_html=True)
            
        with col_diag2:
            st.markdown("<div style='text-align: center; font-weight: bold; margin-bottom: 0.5rem; color: #111827;'>B. General Circular Geometry (N Sessions)</div>", unsafe_allow_html=True)
            st.markdown("""<div style="display: flex; justify-content: center; background-color: #FAFAFA; padding: 1.5rem; border: 2px solid #E5E7EB; border-radius: 8px;">
<svg width="200" height="170" viewBox="0 0 200 170" xmlns="http://www.w3.org/2000/svg">
<!-- Circular track (dashed) -->
<circle cx="100" cy="85" r="60" fill="none" stroke="#FF4B4B" stroke-width="2.5" stroke-dasharray="5,5" />
<!-- Radius lines (dashed) -->
<line x1="100" y1="85" x2="100" y2="25" stroke="#4B5563" stroke-width="1.5" stroke-dasharray="4,4" />
<line x1="100" y1="85" x2="157" y2="66" stroke="#4B5563" stroke-width="1.5" stroke-dasharray="4,4" />
<line x1="100" y1="85" x2="135" y2="134" stroke="#4B5563" stroke-width="1.5" stroke-dasharray="4,4" />
<line x1="100" y1="85" x2="65" y2="134" stroke="#4B5563" stroke-width="1.5" stroke-dasharray="4,4" />
<line x1="100" y1="85" x2="43" y2="66" stroke="#4B5563" stroke-width="1.5" stroke-dasharray="4,4" />
<!-- Center Node A -->
<circle cx="100" cy="85" r="9" fill="#FFFFFF" stroke="#111827" stroke-width="3" />
<text x="100" y="88.5" font-family="'Inter', sans-serif" font-size="10" font-weight="900" text-anchor="middle" fill="#111827">A</text>
<!-- Perimeter Nodes B1 to Bn -->
<circle cx="100" cy="25" r="9" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="3" />
<text x="100" y="28.5" font-family="'Inter', sans-serif" font-size="9" font-weight="900" text-anchor="middle" fill="#FF4B4B">B1</text>

<circle cx="157" cy="66" r="9" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="3" />
<text x="157" y="69.5" font-family="'Inter', sans-serif" font-size="9" font-weight="900" text-anchor="middle" fill="#FF4B4B">B2</text>

<circle cx="135" cy="134" r="9" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="3" />
<text x="135" y="137.5" font-family="'Inter', sans-serif" font-size="9" font-weight="900" text-anchor="middle" fill="#FF4B4B">B3</text>

<circle cx="65" cy="134" r="9" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="3" />
<text x="65" y="137.5" font-family="'Inter', sans-serif" font-size="9" font-weight="900" text-anchor="middle" fill="#FF4B4B">B4</text>

<circle cx="43" cy="66" r="9" fill="#FFFFFF" stroke="#FF4B4B" stroke-width="3" />
<text x="43" y="69.5" font-family="'Inter', sans-serif" font-size="9" font-weight="900" text-anchor="middle" fill="#FF4B4B">Bn</text>
<!-- Radius Text -->
<text x="100" y="160" font-family="'Inter', sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#4B5563">Radius (r)</text>
</svg>
</div>""", unsafe_allow_html=True)
            
        st.markdown(r"""
            ##### B. Parameter Akuisisi Data Lapangan
            *   **Array Radius ($r$):** $r$ (radius of the perimeter geophone stations from the center).
            *   **Recording Duration:** minimum of 20 minutes for each station pair (total duration scales with the number of sessions).
            *   **Sampling Rate:** 100 Hz (retrieved directly from the metadata/header of the uploaded .sac/.mseed seismic records).
            *   **Geophone Sensor:** Geophone (vertical component).
            
            ##### C. Asumsi Kunci
            This method relies heavily on the assumption that the ambient noise wavefield is stationary (its statistical properties remain stable and unchanged) throughout the total duration of all recording sessions. The azimuthal average is simulated numerically during data processing by averaging the cross-coherence results from all station pairs recorded separately.
        """)
        
    elif topik == "4. Data Processing Workflow (Step-by-Step)":
        st.markdown(r"""
            #### 4. Data Processing Workflow (Step-by-Step)
            
            The sequential SPAC microtremor data processing workflow from raw files to the final dispersion curve consists of the following steps:
            
            1.  **Seismogram Data Input:**
                Read the raw binary seismic records in SAC or MiniSEED format for the reference station (A) and perimeter stations (B) across all measurement sessions (Session 1, Session 2, ..., Session N).
            2.  **Temporal Segmentation (Windowing):**
                Continuous long-duration signals are segmented into shorter time windows (e.g., window length of $2^{13} = 8192$ samples or equivalent to 81.92 seconds at a 100 Hz sampling rate) with a specific overlap. This is crucial to isolate stationary data segments and remove local transient noise (such as vibrations from nearby vehicular traffic).
            3.  **Spectral Transformation (FFT):**
                Each time window of time-domain data is transformed into the frequency domain using the Fast Fourier Transform (FFT) algorithm to obtain the amplitude and phase spectra of the signals.
            4.  **Complex Coherence Calculation:**
                Complex coherence is calculated at each frequency for every time window of each station pair session ($A-B_1$, $A-B_2$, ..., $A-B_n$).
            5.  **Quality Control (QC) & Auto-Suggestion of Windows:**
                *   **RMSE Evaluation:** The application computes the Root Mean Square Error (RMSE) between the coherence curve of each window and the median coherence curve over a control frequency range (default 1 - 50 Hz).
                *   **Window Suggestions:** Windows with an RMSE below a cutoff threshold (e.g., median + 0.05 standard deviation) are classified as clean (consistent) windows, while windows with high RMSE are classified as outliers (noise) and suggested for exclusion.
            6.  **Two-Stage Stacking (Averaging):**
                *   *Stage 1:* The selected windows within each session are averaged independently to obtain the average SPAC coefficient curve for each session ($\rho_1, \rho_2, \dots, \rho_N$).
                *   *Stage 2:* All average session curves are then averaged together to produce the final stacked SPAC coefficient curve ($\rho_{avg}$). This is essential to ensure balanced spatial weighting from all azimuthal directions of the sensors.
            7.  **Bessel Function Fitting:**
                The final stacked SPAC coefficient values $\rho_{avg}(f)$ are fitted to the theoretical Bessel function $J_0(x)$ using the least-squares fitting method to determine the Bessel argument $x = kr$ at each frequency.
            8.  **Dispersion Curve Extraction:**
                The Rayleigh wave phase velocity is calculated using the formula $c(f) = \frac{2\pi f r}{x}$. The results are subsequently filtered within specified minimum and maximum phase velocity limits and can be truncated (*frequency cut*) over frequency ranges free of noise.
        """)
        
    elif topik == "5. Supporting Literature & Scientific Journals":
        st.markdown(r"""
            #### 5. Supporting Literature & Scientific Journals
            
            Below is the bibliography of key scientific articles supporting the sequential SPAC method:
            
            1.  Aki, K. (1957). Space and time spectra of stationary stochastic waves, with special reference to microtremors. *Bulletin of the Earthquake Research Institute*, 35, 415–456.
            2.  Cho, I. (2020). Two-sensor microtremor SPAC method: potential utility of imaginary spectrum components. *Geophysical Journal International*, 220(3), 1735–1747.
            3.  Hayashi, K., Asten, M. W., Stephenson, W. J., Cornou, C., Hobiger, M., Pilz, M., & Yamanaka, H. (2022). Microtremor array method using spatial autocorrelation analysis of Rayleigh-wave data. *Journal of Seismology*, 26(4), 601–627.
            4.  Okada, H. (2003). *The Microtremor Survey Method* (K. Suto, Trans.). Society of Exploration Geophysicists.
            5.  Syamsuddin, E., Arif, A.F., Makhrani, Arsyad, A., & Effendi, R. (2025). Spatial Autocorrelation Method (SPAC) for Subsurface Shear Wave Velocity Profiling: A Non-invasive Approach for Site Characterization. *IOP Conference Series: Earth and Environmental Science*, 1525, 012017.
        """)
        
    elif topik == "6. Acknowledgments":
        st.markdown(r'''
            #### 6. Acknowledgments
            
            I (Ahmad Rianul Qauliah, Geophysics 2020, Hasanuddin University), as the developer of this interactive web application, express my appreciation and gratitude to Ahmad Fauzy (Geophysics 2018, Hasanuddin University).
            
            The foundational processing framework of the SPAC method published on his GitHub repository served as a key basis for powering this interactive web application.
            
        ''')

# ==========================================
# TAB 1: INPUT DATA
# ==========================================
with tab_input:
    st.subheader("1. Upload Waveform Files (.sac / .mseed)")
    st.write("Configure the number of measurement sessions (station pairs), then upload the center geophone (A) and perimeter geophone (B) files for each session:")

    # Input for number of sessions
    if "n_sessions_input" not in st.session_state:
        st.session_state.n_sessions_input = 3

    n_sessions = st.number_input(
        "Number of Measurement Sessions (Number of Station Pairs)",
        min_value=1,
        max_value=20,
        value=st.session_state.n_sessions_input,
        step=1
    )
    st.session_state.n_sessions_input = n_sessions

    # Dynamic grid of uploaders (3 columns per row)
    for i in range(0, n_sessions, 3):
        cols = st.columns(min(3, n_sessions - i))
        for j, col in enumerate(cols):
            idx = i + j + 1
            with col:
                st.markdown(f"### Session {idx} (A - B{idx})")
                st.markdown("<hr style='border: 0; height: 2px; background: linear-gradient(to right, #FF4B4B, rgba(255, 75, 75, 0.2)); margin-top: 4px; margin-bottom: 16px;' />", unsafe_allow_html=True)
                
                file_a = st.file_uploader(f"Upload Center A - Session {idx}", type=["sac", "mseed", "miniseed", "msd", "txt", "asc", "dat", "csv"], key=f"file_a{idx}")
                sel_a_idx = 0
                if file_a:
                     stream_a = load_uploaded_file_as_stream(file_a, fs)
                     if stream_a:
                        if len(stream_a) > 1:
                            options_a = [f"Trace {k}: {tr.id} ({tr.stats.npts} pts)" for k, tr in enumerate(stream_a)]
                            selected_a = st.selectbox(
                                f"Select Channel A - Session {idx}",
                                options=options_a,
                                key=f"trace_a_sel{idx}"
                            )
                            sel_a_idx = options_a.index(selected_a)
                        
                        with st.expander("Preview Waveform A", expanded=False):
                            plot_trace_preview(stream_a[sel_a_idx], f"Signal A (Session {idx})")
                
                file_b = st.file_uploader(f"Upload Perimeter B{idx} - Session {idx}", type=["sac", "mseed", "miniseed", "msd", "txt", "asc", "dat", "csv"], key=f"file_b{idx}")
                sel_b_idx = 0
                if file_b:
                     stream_b = load_uploaded_file_as_stream(file_b, fs)
                     if stream_b:
                        if len(stream_b) > 1:
                            options_b = [f"Trace {k}: {tr.id} ({tr.stats.npts} pts)" for k, tr in enumerate(stream_b)]
                            selected_b = st.selectbox(
                                f"Select Channel B{idx} - Session {idx}",
                                options=options_b,
                                key=f"trace_b_sel{idx}"
                            )
                            sel_b_idx = options_b.index(selected_b)
                        
                        with st.expander(f"Preview Waveform B{idx}", expanded=False):
                            plot_trace_preview(stream_b[sel_b_idx], f"Signal B{idx} (Session {idx})")

    st.markdown("---")
    
    # Demo Data Option
    st.write("**Don't have field data yet?** Load our microtremor demo data to try it out:")
    demo_btn = st.button("Load Demo Data")
    
    if demo_btn:
        # We will copy demo data from local demo_data folder in the repo
        demo_src_dir = "./demo_data"
        if os.path.exists(demo_src_dir):
            try:
                for demo_file in ["A41.sac", "B41.sac", "A42.sac", "B42.sac", "A43.sac", "B43.sac"]:
                    src_path = os.path.join(demo_src_dir, demo_file)
                    dest_path = os.path.join(main_folder.path_data, demo_file)
                    shutil.copy(src_path, dest_path)
                st.session_state.n_sessions_input = 3
                st.session_state.demo_loaded = True
                st.session_state.data_list_by_pair = {
                    "Pair_1": ["A41.sac", "B41.sac"],
                    "Pair_2": ["A42.sac", "B42.sac"],
                    "Pair_3": ["A43.sac", "B43.sac"]
                }
                st.success("Demo Data successfully loaded! Open the 'Processing & Quality Control' tab to start processing.")
                st.rerun()
            except Exception as ex:
                st.error(f"Failed to copy demo data: {ex}")
        else:
            st.warning("Demo data folder not found. Please upload your files manually.")

    # Save Uploaded Files
    # Check if user has uploaded any files, if so, only use the uploaded files
    any_uploaded = any(
        st.session_state.get(f"file_a{idx}") is not None or st.session_state.get(f"file_b{idx}") is not None
        for idx in range(1, n_sessions + 1)
    )

    if any_uploaded:
        pairs_to_process = {}
        for idx in range(1, n_sessions + 1):
            file_a = st.session_state.get(f"file_a{idx}")
            file_b = st.session_state.get(f"file_b{idx}")
            if file_a and file_b:
                stream_a = load_uploaded_file_as_stream(file_a, fs)
                stream_b = load_uploaded_file_as_stream(file_b, fs)
                
                if stream_a and stream_b:
                    # Get selected index for A
                    sel_a_idx = 0
                    if len(stream_a) > 1:
                        sel_option_a = st.session_state.get(f"trace_a_sel{idx}")
                        if sel_option_a:
                            sel_a_idx = int(sel_option_a.split(":")[0].split()[-1])
                    
                    # Get selected index for B
                    sel_b_idx = 0
                    if len(stream_b) > 1:
                        sel_option_b = st.session_state.get(f"trace_b_sel{idx}")
                        if sel_option_b:
                            sel_b_idx = int(sel_option_b.split(":")[0].split()[-1])
                    
                    # Generate standardized names
                    name_a = f"sesi_{idx}_A.sac"
                    name_b = f"sesi_{idx}_B.sac"
                    
                    dest_path_a = os.path.join(main_folder.path_data, name_a)
                    dest_path_b = os.path.join(main_folder.path_data, name_b)
                    
                    # Write trace A as SAC
                    tr_a = stream_a[sel_a_idx]
                    if not tr_a.stats.network: tr_a.stats.network = 'XX'
                    if not tr_a.stats.station: tr_a.stats.station = f'S{idx}A'
                    if not tr_a.stats.channel: tr_a.stats.channel = 'Z'
                    tr_a.stats.sampling_rate = fs
                    tr_a.write(dest_path_a, format="SAC")
                    
                    # Write trace B as SAC
                    tr_b = stream_b[sel_b_idx]
                    if not tr_b.stats.network: tr_b.stats.network = 'XX'
                    if not tr_b.stats.station: tr_b.stats.station = f'S{idx}B'
                    if not tr_b.stats.channel: tr_b.stats.channel = 'Z'
                    tr_b.stats.sampling_rate = fs
                    tr_b.write(dest_path_b, format="SAC")
                    
                    pairs_to_process[f"Pair_{idx}"] = [name_a, name_b]
        st.session_state.data_list_by_pair = pairs_to_process

    # Display status
    if len(st.session_state.data_list_by_pair) > 0:
        st.info("Files ready for processing: " + ", ".join([f"{k}: {v[0]} & {v[1]}" for k, v in st.session_state.data_list_by_pair.items()]))
    else:
        st.warning("Please upload a dataset containing at least 1 session or load the Demo Data.")

# ==========================================
# TAB 2: PROSES & QUALITY CONTROL
# ==========================================
with tab_processing:
    st.subheader("2. Data Processing & Window Stacking (Quality Control)")
    
    if len(st.session_state.data_list_by_pair) == 0:
        st.warning("Please upload your data or load the Demo Data in the Data Input & Sessions tab before processing.")
    else:
        # Run process button
        btn_run = st.button("Start Windowing & Coherence Processing", type="primary")
        
        if btn_run or st.session_state.coherence_calculated:
            with st.spinner("Processing signals, segmenting windows, and computing coherence..."):
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
                        st.markdown(f"#### Processing Session {pair_key} ({files[0]} - {files[1]})")
                        
                        # Read data
                        data_init = readdata.ReadData(main_folder, files, 'SAC')
                        datas = data_init.read_data()
                        
                        # Show windowed data plots side-by-side
                        st.markdown(f"##### Windowed Data Visualization - Session {pair_key}")
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
                    st.success("Waveform processing complete! Please perform Quality Control below.")
                    
                    # Store computed values in session state for dynamic UI update
                    st.session_state.windowing_objs = windowing_objs
                    st.session_state.last_spac_coef_obj = last_spac_coef_obj
                    st.session_state.n_windows_found = n_windows_found

                except Exception as ex:
                    st.error(f"An error occurred while processing data: {ex}")
                    st.session_state.coherence_calculated = False

        # --- Quality Control Section (Dynamic UI) ---
        if st.session_state.coherence_calculated:
            st.markdown("---")
            st.markdown("### Quality Control: Coherence Window Selection")
            st.write("Visualize all coherence windows and use the auto-suggestion feature to filter consistent/similar data:")

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
            st.write("Click the expanders below to view the coherence curves for each window:")
            for pair_name, wins in sorted(pairs_win.items()):
                with st.expander(f"Show Coherence Grid Plot - Session {pair_name}", expanded=False):
                    plot_coherence_grid_for_pair(pair_name, wins, os.path.join(working_dir, "processing"))

            st.write("Configure filter parameters and select windows for stacking below:")
            
            # Parameter Saran Window Otomatis moved outside st.form so that adjustments trigger real-time updates
            st.write("**Auto-Suggestion Window Parameters**")
            col_param1, col_param2, col_param3 = st.columns(3)
            with col_param1:
                std_threshold = st.slider("Selection Tightness (std multiplier)", min_value=0.01, max_value=0.50, value=0.05, step=0.01)
            with col_param2:
                f_min_check = st.number_input("Min Frequency Range (Hz)", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
            with col_param3:
                f_max_check = st.number_input("Max Frequency Range (Hz)", min_value=1.0, max_value=50.0, value=50.0, step=0.5)

            with st.form("qc_selection_form"):
                st.markdown("---")
                selected_pairs_win = {}
                
                # Group selections by pair
                for pair_name, wins in sorted(pairs_win.items()):
                    st.markdown(f"#### Analysis & Selection: Session {pair_name}")
                    
                    # 2. Get similarity suggestions
                    suggested_wins, outlier_wins = suggest_similar_windows(
                        pair_name, wins, os.path.join(working_dir, "processing"),
                        std_threshold=std_threshold, f_min_check=f_min_check, f_max_check=f_max_check
                    )
                    
                    # 3. Display suggestions
                    col_sug1, col_sug2 = st.columns(2)
                    with col_sug1:
                        st.success(f"**Consistent Window Suggestions**: {suggested_wins}")
                    with col_sug2:
                        if outlier_wins:
                            st.warning(f"**Outlier Windows (Not Recommended)**: {outlier_wins}")
                        else:
                            st.info("**Outlier Windows**: No outliers detected.")
                    
                    # 4. Multiselect widget for this pair, default to suggested_wins
                    selected_wins = st.multiselect(
                        f"Select window numbers for stacking ({pair_name}):",
                        options=sorted([int(w) for w in wins]),
                        default=suggested_wins,
                        key=f"select_{pair_name}"
                    )
                    selected_pairs_win[pair_name] = selected_wins
                    st.markdown("---")
                
                # Submit button
                submit_qc = st.form_submit_button("Apply Selection & Calculate Stacking", type="primary")

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
                st.error("Please select at least 1 clean window in one of the sessions to perform Stacking.")
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
                st.markdown("### Final Stacked SPAC Coefficient Results")
                st.write("The curve below represents the two-stage average (average per session, then averaged across sessions):")
                
                # Plotly Plot for observed SPAC coefficient
                fig_spac = go.Figure()
                fig_spac.add_trace(go.Scatter(
                    x=freq,
                    y=avspac,
                    mode='lines+markers',
                    name='Two-Stage Average (Stacked)',
                    line=dict(color='black', width=2),
                    marker=dict(size=4)
                ))
                
                # Optional: Plot individual pair averages in distinct colors for comparison
                colors_list = px.colors.qualitative.Plotly # Plotly sequence has 10 distinct colors
                for idx_pair, (pair_name, pair_avg) in enumerate(sorted(pair_averages.items())):
                    color = colors_list[idx_pair % len(colors_list)]
                    fig_spac.add_trace(go.Scatter(
                        x=freq,
                        y=pair_avg,
                        mode='lines',
                        name=f'Session {pair_name} Average',
                        line=dict(color=color, width=1.5, dash='dash')
                    ))
                
                fig_spac.update_layout(
                     title=dict(
                         text="Final SPAC Coefficient vs. Frequency",
                         font=dict(color="black")
                     ),
                     xaxis=dict(
                         title=dict(text="Frequency (Hz)", font=dict(color="black")),
                         range=[0, 50],
                         showgrid=True,
                         gridcolor="lightgray",
                         linecolor="black",
                         ticks="outside",
                         tickcolor="black",
                         tickfont=dict(color="black")
                     ),
                     yaxis=dict(
                         title=dict(text="SPAC Coefficient", font=dict(color="black")),
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
                
                st.success("Average SPAC coefficient successfully stacked! Open the Dispersion Curve tab to view the fitting results.")

# ==========================================
# TAB 3: DISPERSION CURVE FINAL
# ==========================================
with tab_output:
    st.subheader("3. Dispersion Curve (Bessel J0 Function Fitting)")
    
    if "spaccoeff_obj" not in st.session_state:
        st.warning("Please complete the processing and Quality Control in the Processing & QC tab first.")
    else:
        st.info(f"Calculating dispersion curve with phase velocity bounds: {min_pv} m/s to {max_pv} m/s.")
        
        # Inisialisasi Objek DispersionCurve
        dc = dispersioncurve.DispersionCurve(st.session_state.spaccoeff_obj)
        
        # Decimator
        # decimator reduces samples count to speed up the least-square fitting
        freq_dec, avspac_dec = dc.decimator(decimation_factor)
        
        # Calculate dispersion curve
        with st.spinner("Fitting SPAC coefficient to Bessel function (Least-Squares Fitting)..."):
            try:
                # returns: pv, f_pv, bessel_fit, x_bessel, fitted_avspac
                pv, f_pv, bessel_fit, x_bessel, fitted_avspac = dc.calculate_dispcurv(min_pv, max_pv)
                
                # --- Frequency Cut Feature ---
                st.markdown("### Dispersion Curve Frequency Range Truncation")
                st.write("Adjust the frequency range to use for further analysis:")
                with st.form("form_frequency_cut"):
                    col_cut1, col_cut2 = st.columns(2)
                    with col_cut1:
                        f_min_val = float(np.round(np.min(f_pv), 2))
                        cut_f_min = st.number_input("Minimum Frequency Cut (Hz)", min_value=0.0, max_value=100.0, value=max(0.0, f_min_val), step=0.1)
                    with col_cut2:
                        f_max_val = float(np.round(np.max(f_pv), 2))
                        cut_f_max = st.number_input("Maximum Frequency Cut (Hz)", min_value=0.0, max_value=100.0, value=min(50.0, f_max_val), step=0.1)
                    st.form_submit_button("Apply Truncation", type="primary")
                
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
                    st.markdown("#### Bessel J0 Function Fitting")
                    
                    fig_bessel = go.Figure()
                    
                    # Ideal J0 Curve
                    x_ideal = np.linspace(0, 15, 200)
                    y_ideal = jv(0, x_ideal)
                    fig_bessel.add_trace(go.Scatter(
                        x=x_ideal,
                        y=y_ideal,
                        mode='lines',
                        name='Ideal Bessel J0 Function',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Observed Data points plotted at the calculated Bessel coordinates (Show all points in gray, active in black)
                    fig_bessel.add_trace(go.Scatter(
                        x=x_bessel,
                        y=avspac_dec,
                        mode='markers',
                        name='Field Data (Out of Range)',
                        marker=dict(color='lightgray', size=5),
                        showlegend=True
                    ))
                    
                    fig_bessel.add_trace(go.Scatter(
                        x=x_bessel_cut,
                        y=avspac_dec_cut,
                        mode='markers',
                        name='Field Data (Active)',
                        marker=dict(color='black', size=7)
                    ))
                    
                    fig_bessel.update_layout(
                        title=dict(
                            text="Field Data Fitting to Bessel J0 Function",
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
                            title=dict(text="SPAC Coefficient", font=dict(color="black")),
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
                    st.markdown("#### Rayleigh Dispersion Curve")
                    
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
                        name='All Frequencies',
                        line=dict(color='lightgray', width=1),
                        marker=dict(color='lightgray', size=4)
                    ))
                    
                    # Cut Curve (Blue)
                    fig_disp.add_trace(go.Scatter(
                        x=f_pv_cut,
                        y=pv_cut,
                        mode='lines+markers',
                        name='Truncated Range (Active)',
                        line=dict(color='blue', width=2.5),
                        marker=dict(color='darkblue', size=6)
                    ))
                    
                    fig_disp.update_layout(
                         title=dict(
                             text="Rayleigh Dispersion Curve (Phase Velocity vs. Frequency)",
                             font=dict(color="black")
                         ),
                         xaxis=dict(
                             title=dict(text="Frequency (Hz)", font=dict(color="black")),
                             range=x_range,
                             showgrid=True,
                             gridcolor="lightgray",
                             linecolor="black",
                             ticks="outside",
                             tickcolor="black",
                             tickfont=dict(color="black")
                         ),
                         yaxis=dict(
                             title=dict(text="Phase Velocity (m/s)", font=dict(color="black")),
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
                st.markdown("### Export Analysis Results")
                
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
                st.write("Preview of Truncated Dispersion Curve Data (Active):")
                st.dataframe(export_df_cut, use_container_width=True)
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    csv_full = export_df_full.to_csv(index=False)
                    st.download_button(
                        label="Download Full Dispersion Curve (CSV)",
                        data=csv_full,
                        file_name=f"dispersion_curve_FULL_radius_{radius}m.csv",
                        mime="text/csv",
                        key="btn_full"
                    )
                with col_btn2:
                    csv_cut = export_df_cut.to_csv(index=False)
                    st.download_button(
                        label="Download Truncated Dispersion Curve (CSV)",
                        data=csv_cut,
                        file_name=f"dispersion_curve_CUT_{cut_f_min:.1f}to{cut_f_max:.1f}Hz.csv",
                        mime="text/csv",
                        key="btn_cut"
                    )
                

                
            except Exception as ex_fit:
                st.error(f"Failed to perform least-squares fitting: {ex_fit}")

# ==========================================
# TAB 5: INVERSI VS 1D
