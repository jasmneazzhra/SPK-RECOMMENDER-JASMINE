# **SPK Rekomendasi Multi-Metode (General CSV Recommendation System)**

Sistem Pendukung Keputusan berbasis kombinasi beberapa metode rekomendasi untuk berbagai jenis dataset CSV.

---

## **1. Deskripsi Project**

Project ini merupakan **Sistem Pendukung Keputusan (SPK)** yang dapat menerima *file CSV apa saja*, membaca isi dataset tersebut, menampilkan ringkasan data, dan memberikan rekomendasi menggunakan beberapa metode sekaligus, yaitu:

* Content-Based Filtering (TF-IDF similarity dari kolom teks)
* Case-Based Reasoning (similarity berdasarkan kolom numerik)
* Clustering (menggunakan K-Means)
* Hybrid Scoring (penggabungan semua metode dengan bobot tertentu)
* Chatbot sederhana yang dapat mendeteksi judul dari pertanyaan pengguna dan memberikan rekomendasi otomatis

Sistem dibangun menggunakan Streamlit agar dapat diakses secara mudah melalui web.

---

## **2. Fitur Utama**

### **2.1 Upload CSV Umum**

Pengguna dapat mengunggah dataset apa pun. Sistem akan menampilkan:

* Jumlah baris dan kolom
* Nama kolom
* Tipe data
* Preview dataset

Data dibaca menggunakan pandas.

### **2.2 Pemilihan Kolom**

Pengguna dapat memilih:

* Kolom identifier (misalnya: title, name)
* Kolom teks untuk TF-IDF
* Kolom numerik untuk processing CBR

### **2.3 Sistem Rekomendasi Multi-Metode**

Terdapat tiga metode inti:

1. **Content-Based (TF-IDF Similarity)**
2. **CBR Numeric Similarity (StandardScaler + Euclidean Distance)**
3. **Cluster Similarity (K-Means Label Score)**

Ketiganya digabungkan dengan bobot yang dapat disesuaikan oleh pengguna.

### **2.4 Chatbot Rekomendasi**

Pengguna dapat mengetik pertanyaan seperti:

> "Aku habis nonton Breaking Bad, apa rekomendasi film lain?"

Chatbot akan:

* Mencari judul dalam pertanyaan
* Jika cocok, sistem menghasilkan rekomendasi otomatis
* Jika tidak ditemukan, chatbot memberi saran untuk memasukkan judul lebih spesifik

### **2.5 Download Hasil Rekomendasi**

Hasil dapat diunduh dalam format CSV.

---

## **3. Struktur Folder**

```
project/
│
├── app.py
├── requirements.txt
│
├── utils/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── recommenders.py
│   ├── chatbot.py
│
└── README.md
```

---

## **4. Cara Menjalankan**

### 4.1 Install dependencies

```
pip install -r requirements.txt
```

Jika terjadi error pyarrow, gunakan versi pandas yang stabil:

```
pip install pandas==2.2.2
```

### 4.2 Jalankan Streamlit

```
streamlit run app.py
```

Aplikasi akan terbuka melalui browser.

---

## **5. Alur Kerja Sistem**

### **5.1 Pembacaan Data**

* Ketika file CSV diunggah, pandas langsung membaca seluruh baris, kolom, dan tipe data.
* Sistem menampilkan preview agar pengguna mengetahui isi dataset.

### **5.2 Preprocessing**

* Kolom teks → diproses menggunakan TF-IDF Vectorizer
* Kolom numerik → distandarisasi menggunakan StandardScaler
* Clustering → KMeans membentuk label cluster

### **5.3 Perhitungan Rekomendasi**

Setiap item diberi skor melalui rumus hybrid:

```
Final_Score = (w_text * sim_text) + (w_num * sim_numeric) + (w_cluster * sim_cluster)
```

Item diurutkan berdasarkan skor tertinggi, lalu ditampilkan sebagai rekomendasi.

### **5.4 Alur Kerja Chatbot**

1. Pengguna bertanya dalam bahasa bebas.
2. Sistem mengekstrak kemungkinan judul dari kalimat (mencocokkan dengan kolom identifier).
3. Jika match ditemukan → mesin rekomendasi dipanggil.
4. Jika tidak ada → chatbot memberi peringatan bahwa judul tidak ditemukan.

---

## **6. Contoh Penggunaan**

1. Upload CSV berisi film
2. Pilih kolom `title` sebagai identifier
3. Pilih kolom teks `overview` untuk TF-IDF
4. Pilih kolom numerik seperti `rating` atau `popularity`
5. Klik *Proses Model*
6. Masukkan pertanyaan seperti:
   "Setelah menonton Interstellar, film apa yang mirip?"

---

## **7. Koordinator dan Pengembang**

Disusun oleh Jasmine Az Zahra Ihsani
Program Studi Teknologi Rekayasa Perangkat Lunak
Politeknik Negeri Banyuwangi
