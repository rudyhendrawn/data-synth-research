# Studi Ablasi

## 1. Mengapa studi ablation penting di penelitian ini?

- **Memisahkan kontribusi tiap komponen**  
  Kita menguji apa yang benar-benar memberi dampak: oversampling (SMOTE/CTGAN), arsitektur model (XGBoost vs GNN), sinyal anomali (AE/IF/LOF), kalibrasi (Platt/Temperature/Isotonic), sampai jenis fitur/edge graf. Dengan menghapus/menonaktifkan satu faktor pada satu waktu, kita bisa mengukur sumbangan bersihnya terhadap PR-AUC/Recall/F1. Ini definisi klasik ablation di ML.

- **Membuktikan bahwa "lift" bukan artefak evaluasi**  
  Pada data sangat tidak seimbang, PR-AUC/precision–recall lebih tepat daripada ROC; ablation memastikan kenaikan metrik bukan akibat metrik yang bias atau prosedur evaluasi yang salah.

- **Menghindari kesimpulan palsu karena leakage waktu**  
  Dengan purged/embargo split (forward-chaining + gap waktu), ablation memperlihatkan bahwa perbaikan tetap ada meski protokol anti-leakage diterapkan. Ini penting di domain finansial/time-series.

- **Transparansi/akuntabilitas metodologi**  
  Reviewer ingin tahu mengapa kombinasi tertentu unggul. Ablation yang rapi (mis. "SMOTE-ENN memberi +ΔPR-AUC di XGBoost, tetapi tidak di GNN") memperkuat klaim ilmiah dan interpretabilitas praktik.

## 2. Bagaimana melakukan studi ablation (langkah demi langkah)

Prinsip umum: ubah satu hal pada satu waktu (single-factor), pakai split yang sama, dan laporkan mean ± SD (≥5 seeds) serta CI bootstrap bila perlu.

### 2.1 Rancang hipotesis & baseline

Tetapkan baseline: tanpa oversampling, model terawasi tabular (mis. XGBoost), tanpa sinyal anomali, tanpa kalibrasi.

Definisikan hipotesis contoh:

- H1: SMOTE menaikkan PR-AUC/Recall dibanding None (train-only).
- H2: CTGAN (tabular GAN) menambah lift di dataset dengan fitur campuran dibanding SMOTE.
- H3: GNN mengungguli model tabular saat relasi antar entitas relevan.
- H4: Kalibrasi (Temperature/Platt/Isotonic) memperbaiki ECE/Brier dan menstabilkan Recall@Precision≥P₀.

### 2.2 Kunci protokol anti-bias

Split waktu (forward-chaining) + gap (purged/embargo); dilarang mengacak waktu. Gunakan protokol López de Prado untuk mencegah temporal leakage.

Oversampling hanya di train (bukan di val/test). Untuk SMOTE: rujukan asli JAIR; untuk CTGAN: NeurIPS 2019.

Metrik utama: PR-AUC, Recall, F1 (kelas fraud); gunakan PR curve karena imbalance.

### 2.3 Daftar ablation yang relevan (susun sebagai matriks eksperimen)

- **Oversampling vs None (satu model tetap)**  
  Tetap pakai XGBoost dengan hyperparameter konstan. Bandingkan: None / ROS / SMOTE / Borderline-SMOTE / SMOTE-ENN / CTGAN. Tujuan: isolasi efek penyeimbangan.

- **Model vs model (satu skema data tetap)**  
  Tanpa mengganti fitur/split, bandingkan LR/DT/XGBoost/MLP vs GNN (GCN/GraphSAGE/GAT) pada data yang sama (tabular vs graf terbangun). Tujuan: kontribusi arsitektur relasional.

- **Ablasi graf (khusus GNN)**  
  - Jenis edge: hapus satu jenis relasi (mis. perangkat bersama, IP bersama, merchant bersama) → ukur Δ-metrik.  
  - Kedalaman tetangga (1-hop vs 2-hop) dan mekanisme attention (on/off untuk GAT). Tujuan: komponen graf mana yang membawa sinyal.  
  (Tetap evaluasi dengan PR-AUC/Recall karena imbalance.)

- **Sinyal anomali (unsupervised)**  
  Early fusion (skor AE/IF/LOF sebagai fitur) vs late fusion (kombinasi skor di akhir) vs "tanpa anomali". Tujuan: nilai tambah sinyal unsupervised terhadap supervised.

- **Kalibrasi probabilitas**  
  Bandingkan tanpa kalibrasi vs Temperature Scaling (1 parameter) vs Platt Scaling (sigmoid) vs Isotonic Regression (non-parametrik). Ukur ECE/Brier + Recall@Precision≥P₀ di test. (Kalibrasi fit di validation saja).

- **Fitur/rekayasa fitur**  
  Grup fitur waktu (jendela 7/30 hari), perilaku perangkat, merchant risk, dll. Ablasi per grup (on/off) untuk melihat fitur yang paling kontributif (hati-hati agar tak menimbulkan leakage—semua agregasi hanya hingga timestamp observasi).

### 2.4 Eksekusi & pelaporan

Kontrol variabel: saat menguji satu faktor, jangan ubah faktor lain (fitur, split, hyperparameter).

Ulangi tiap percobaan dengan ≥5 random seeds; laporkan mean ± SD.

CI bootstrap untuk PR-AUC/Recall (opsional) agar perbedaan signifikan dapat terlihat.

Tabel ablation: sel berisi ΔPR-AUC/ΔRecall/ΔF1 vs baseline; beri bold pada peningkatan yang konsisten.

## 3. Contoh rancangan ablation yang bisa langsung dipakai

- **Ablasi 1 — Oversampling pada XGBoost (IEEE-CIS)**  
  Baseline: None → uji ROS/SMOTE/Borderline/SMOTE-ENN/CTGAN.  
  Harapan: SMOTE/SMOTE-ENN naikkan Recall/F1; CTGAN potensial unggul jika fitur diskret-kontinu campur.

- **Ablasi 2 — Arsitektur pada data graf**  
  Bangun graf (akun–perangkat–merchant). Bandingkan XGBoost (fitur tabular) vs GCN/GraphSAGE/GAT.  
  Harapan: GNN unggul jika pola kolusi/relay kuat (edge/2-hop penting). (PR-AUC acuan evaluasi).

- **Ablasi 3 — Sinyal anomali**  
  XGBoost terbaik dari Ablasi 1: tambahkan skor Isolation Forest/LOF/AE sebagai fitur (early) dan sebagai skor fusi (late).  
  Harapan: Recall@Precision≥P₀ naik pada kasus baru (novel).

- **Ablasi 4 — Kalibrasi**  
  Model terbaik (dari 1–3): bandingkan No-Cal vs TS vs Platt vs Isotonic.  
  Harapan: ECE/Brier membaik; Recall@Precision≥95% meningkat/lebih stabil setelah kalibrasi (ingat PR lebih tepat untuk imbalance).

- **Ablasi 5 — Anti-leakage**  
  Ulangi satu eksperimen kunci dengan purged/embargo vs TimeSeriesSplit biasa untuk menunjukkan bahwa temuan tetap valid di protokol yang kuat. 