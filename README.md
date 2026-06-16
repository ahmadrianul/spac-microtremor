# SPAC Microtremor Analysis Web App

Aplikasi web berbasis Streamlit untuk analisis mikrotremor menggunakan metode SPAC (Spatial Autocorrelation) untuk memperoleh kurva dispersi tanah setempat.

## Cara Menjalankan Secara Lokal

1. Buka **Anaconda Prompt**.
2. Masuk ke folder proyek ini:
   ```bash
   cd "C:\Users\RIANO\.gemini\antigravity\scratch\spac-microtremor"
   ```
3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

## Cara Menghubungkan ke GitHub & Streamlit Cloud

1. Buat repositori baru di GitHub dengan nama `spac-microtremor` (jangan centang opsi "Add a README file").
2. Di folder proyek lokal Anda (`C:\Users\RIANO\.gemini\antigravity\scratch\spac-microtremor`), jalankan perintah git berikut:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/USERNAME_ANDA/spac-microtremor.git
   git push -u origin main
   ```
3. Buka halaman Streamlit Cloud yang ada di screenshot Anda.
4. Isi data berikut:
   - **Repository:** `USERNAME_ANDA/spac-microtremor`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
5. Klik **Deploy!**
