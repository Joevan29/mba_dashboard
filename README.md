# 🛒 Market Basket Analysis Dashboard

Proyek ini adalah sebuah *dashboard* analitik interaktif yang dibangun untuk memfasilitasi *Market Basket Analysis* menggunakan **Algoritma Apriori**. Proyek ini dikembangkan sebagai bagian dari tugas akhir/skripsi untuk mengekstraksi wawasan bisnis, pola pembelian pelanggan, dan aturan asosiasi produk pada sistem informasi ritel.

## 🌟 Fitur Utama

Dashboard ini dilengkapi dengan 8 modul analisis utama:

1. **📊 Executive Summary**: Ringkasan level tinggi mengenai dataset ritel, metrik performa algoritma Apriori, dan wawasan bisnis utama (*bundling*, promosi, tata letak).
2. **🔬 Apriori Deep Dive**: Eksplorasi mendalam mengenai aturan asosiasi (Association Rules) yang dihasilkan. Dilengkapi dengan filter interaktif untuk *Support*, *Confidence*, dan *Lift*, serta visualisasi *heatmap*.
3. **🛍️ Business Intelligence**: Menghasilkan strategi bisnis yang dapat ditindaklanjuti, mencakup rekomendasi *cross-selling/up-selling*, optimasi tata letak toko (*store layout*), dan kecerdasan manajemen inventaris berdasarkan perputaran produk.
4. **🤔 Smart Recommender**: Mesin rekomendasi cerdas yang memungkinkan simulasi keranjang belanja untuk memprediksi probabilitas produk selanjutnya yang akan dibeli pelanggan.
5. **⏰ Temporal Analysis**: Analisis pola waktu yang mengungkap kebiasaan belanja pelanggan berdasarkan waktu (jam sibuk, hari kerja vs akhir pekan).
6. **🏬 Department Intelligence**: Analisis level departemen untuk merumuskan strategi dan aturan asosiasi yang lebih spesifik pada departemen tertentu.
7. **🕸️ Visual Analytics**: Visualisasi tingkat lanjut menggunakan *Network Graph* untuk melihat jaringan asosiasi produk, *Sunburst Chart*, dan *Parallel Coordinates*.
8. **📑 Report Generator**: Pembuatan laporan otomatis (Ringkasan Eksekutif, Laporan Teknis, dan Rekomendasi Bisnis) yang dapat diunduh langsung dalam format Markdown dan CSV.

## 🛠️ Teknologi yang Digunakan

Proyek ini dibangun menggunakan bahasa pemrograman Python dengan pustaka-pustaka berikut:
* **[Streamlit](https://streamlit.io/)**: Kerangka kerja utama untuk membangun antarmuka pengguna (UI) dashboard secara interaktif.
* **[MLxtend](http://rasbt.github.io/mlxtend/)**: Pustaka *Machine Learning* untuk pemrosesan dataset transaksi dan implementasi algoritma Apriori.
* **[Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)**: Manipulasi, pembersihan, dan analisis struktur data.
* **[Plotly](https://plotly.com/python/)**: Pembuatan grafik dan visualisasi data yang dinamis dan interaktif.
* **[NetworkX](https://networkx.org/)**: Pembuatan visualisasi jaringan (*network graph*) dari aturan asosiasi.

## ⚙️ Cara Instalasi dan Menjalankan Proyek

Ikuti langkah-langkah berikut untuk menjalankan *dashboard* ini di komputer lokal Anda:

1. **Clone Repositori** (jika menggunakan Git):
   ```bash
   git clone <url-repositori-anda>
   cd mba_dashboard

    Siapkan Virtual Environment (Opsional namun disarankan):
    Bash

    python -m venv env
    source env/bin/activate  # Untuk Linux/Mac
    env\Scripts\activate     # Untuk Windows

    Instal Dependensi:
    Pastikan Anda berada di direktori yang sama dengan file requirements.txt.
    Bash

    pip install -r requirements.txt

    Siapkan Dataset:
    Pastikan file dataset ECommerce_consumer behaviour.csv sudah berada di dalam direktori proyek yang sama dengan file app.py.

    Jalankan Aplikasi:
    Bash

    streamlit run app.py

    Aplikasi akan otomatis terbuka di browser Anda melalui http://localhost:8501.

🎓 Penulis

Joevan NPM: 227006516015

Program Studi Sistem Informasi, Universitas Nasional (UNAS)

Dibuat untuk keperluan Skripsi / Tugas Akhir.
