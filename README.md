# SPAC Microtremor Analysis Web App

An interactive Streamlit-based web application for ambient noise/microtremor analysis using the Spatial Autocorrelation (SPAC) method to extract phase velocity dispersion curves.

*Aplikasi web interaktif berbasis Streamlit untuk analisis ambient noise/mikrotremor menggunakan metode Spatial Autocorrelation (SPAC) untuk mengekstrak kurva dispersi kecepatan fase.*

---

## Language / Bahasa

* [English](#english)
* [Bahasa Indonesia](#bahasa-indonesia)

---

## English

### 1. Web App Explanation
This web application provides a user-friendly and interactive interface to process passive microtremor data. It implements the sequential two-station (or multi-session circular) Spatial Autocorrelation (SPAC) method. Users can dynamically configure the number of sessions, upload raw waveform files (supporting SAC, MiniSEED, and ASCII formats), perform time segmentation (windowing), calculate complex coherence, filter out outliers using auto-suggestion parameters, stack coherence curves, fit them to the zero-order Bessel function ($J_0$), and extract the final Rayleigh wave phase velocity dispersion curve. The results can be visualized interactively and downloaded as CSV files.

### 2. Brief Explanation of SPAC
The Spatial Autocorrelation (SPAC) method, originally formulated by Aki (1957), is a non-invasive passive geophysical technique. It utilizes ambient seismic noise (such as ocean waves, wind, traffic, and industrial vibrations) dominated by surface waves (mainly Rayleigh waves) to characterize subsurface shear wave velocity ($V_s$) structures. By calculating the spatial coherence of microtremors recorded at a center station and several perimeter stations, the average azimuthal correlation is matched to the zero-order Bessel function of the first kind:
$$\rho(f, r) = J_0\left(\frac{2\pi f r}{c(f)}\right)$$
where $r$ is the array radius, $f$ is frequency, and $c(f)$ is the Rayleigh wave phase velocity. Solving this equation yields the dispersion curve ($c(f)$ vs. $f$), which is essential for characterizing soil stiffness and shear wave profiles.

### 3. How to Access and Run the App

#### Local Execution (Anaconda / Command Prompt)
1. Open your terminal or **Anaconda Prompt**.
2. Navigate to the project directory:
   ```bash
   cd "C:\Users\RIANO\.gemini\antigravity\scratch\spac-microtremor"
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit web application:
   ```bash
   streamlit run streamlit_app.py
   ```
5. Your default web browser will automatically open showing the app at `http://localhost:8501`.

#### Cloud Deployment (Streamlit Cloud)
To deploy the application online:
1. Push the project repository to your GitHub account.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New app", select your repository, set the branch to `main`, and enter `streamlit_app.py` as the main file path.
4. Click "Deploy!".

### 4. Supporting Scientific Literature
The processing library and theory implemented in this application are supported by the following literature:
1. **Aki, K. (1957)**. Space and time spectra of stationary stochastic waves, with special reference to microtremors. *Bulletin of the Earthquake Research Institute*, 35, 415–456.
2. **Okada, H. (2003)**. *The Microtremor Survey Method* (K. Suto, Trans.). Society of Exploration Geophysicists.
3. **Cho, I. (2020)**. Two-sensor microtremor SPAC method: potential utility of imaginary spectrum components. *Geophysical Journal International*, 220(3), 1735–1747.
4. **Hayashi, K. et al. (2022)**. Microtremor array method using spatial autocorrelation analysis of Rayleigh-wave data. *Journal of Seismology*, 26(4), 601–627.
5. **Syamsuddin, E. et al. (2025)**. Spatial Autocorrelation Method (SPAC) for Subsurface Shear Wave Velocity Profiling: A Non-invasive Approach for Site Characterization. *IOP Conference Series: Earth and Environmental Science*, 1525, 012017.

---

## Bahasa Indonesia

### 1. Penjelasan Web App
Aplikasi web ini menyediakan antarmuka interaktif yang mudah digunakan untuk memproses data mikrotremor pasif. Aplikasi ini mengimplementasikan metode Spatial Autocorrelation (SPAC) sekuensial dua stasiun (atau lingkaran multi-sesi). Pengguna dapat menentukan jumlah sesi secara dinamis, mengunggah file waveform mentah (mendukung format SAC, MiniSEED, dan ASCII), melakukan segmentasi waktu (windowing), menghitung koherensi kompleks, menyaring outlier menggunakan parameter saran otomatis, melakukan stacking kurva koherensi, mencocokkannya ke fungsi Bessel orde nol ($J_0$), dan mengekstrak kurva dispersi kecepatan fase gelombang Rayleigh. Hasil analisis dapat divisualisasikan secara interaktif dan diunduh dalam format CSV.

### 2. Penjelasan Singkat Metode SPAC
Metode Spatial Autocorrelation (SPAC), yang pertama kali diformulasikan oleh Aki (1957), adalah teknik geofisika pasif non-invasif. Metode ini memanfaatkan kebisingan seismik ambient (seperti gelombang laut, angin, lalu lintas, dan getaran industri) yang didominasi oleh gelombang permukaan (terutama gelombang Rayleigh) untuk mencirikan struktur kecepatan gelombang geser ($V_s$) bawah permukaan. Dengan menghitung koherensi spasial mikrotremor yang direkam di stasiun pusat dan beberapa stasiun keliling, rata-rata korelasi azimuthal dicocokkan dengan fungsi Bessel jenis pertama orde nol:
$$\rho(f, r) = J_0\left(\frac{2\pi f r}{c(f)}\right)$$
di mana $r$ adalah radius lingkaran array, $f$ adalah frekuensi, dan $c(f)$ adalah kecepatan fase gelombang Rayleigh. Pemecahan persamaan ini menghasilkan kurva dispersi ($c(f)$ vs. $f$), yang penting untuk karakterisasi kekakuan tanah dan profil Vs.

### 3. Cara Mengakses dan Menjalankan Aplikasi

#### Menjalankan Secara Lokal (Anaconda / Command Prompt)
1. Buka terminal atau **Anaconda Prompt**.
2. Arahkan ke direktori folder proyek:
   ```bash
   cd "C:\Users\RIANO\.gemini\antigravity\scratch\spac-microtremor"
   ```
3. Instal pustaka dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```
5. Browser web Anda secara otomatis akan terbuka menampilkan aplikasi pada alamat `http://localhost:8501`.

#### Penerbitan Cloud (Streamlit Cloud)
Untuk menayangkan aplikasi secara online agar dapat diakses publik:
1. Unggah folder proyek lokal ini ke repositori akun GitHub Anda.
2. Masuk ke dashboard [Streamlit Community Cloud](https://share.streamlit.io/).
3. Klik "New app", pilih repositori Anda, pilih branch ke `main`, dan isi `streamlit_app.py` pada jalur file utama (main file path).
4. Klik "Deploy!".

### 4. Literatur Ilmiah Pendukung
Pustaka pemrosesan dan teori yang diimplementasikan dalam aplikasi ini didukung oleh artikel-artikel ilmiah berikut:
1. **Aki, K. (1957)**. Space and time spectra of stationary stochastic waves, with special reference to microtremors. *Bulletin of the Earthquake Research Institute*, 35, 415–456.
2. **Okada, H. (2003)**. *The Microtremor Survey Method* (K. Suto, Trans.). Society of Exploration Geophysicists.
3. **Cho, I. (2020)**. Two-sensor microtremor SPAC method: potential utility of imaginary spectrum components. *Geophysical Journal International*, 220(3), 1735–1747.
4. **Hayashi, K. et al. (2022)**. Microtremor array method using spatial autocorrelation analysis of Rayleigh-wave data. *Journal of Seismology*, 26(4), 601–627.
5. **Syamsuddin, E. et al. (2025)**. Spatial Autocorrelation Method (SPAC) for Subsurface Shear Wave Velocity Profiling: A Non-invasive Approach for Site Characterization. *IOP Conference Series: Earth and Environmental Science*, 1525, 012017.
