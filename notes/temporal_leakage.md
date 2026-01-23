# “Gap waktu” & “forward-chaining” untuk mencegah temporal leakage

## Apa itu temporal leakage

**Definisi:** kebocoran informasi karena data masa depan ikut memengaruhi pelatihan/prediksi masa lalu.

**Contoh umum pada fraud:**

- Fitur/normalisasi dihitung memakai seluruh periode (termasuk setelah tanggal prediksi).
- Fitur agregat (mis. count 30 hari) tidak dipotong di tanggal split, sehingga transaksi di Val/Test memengaruhi statistik Train.
- Label lag/chargeback delay: label “fraud” baru dikonfirmasi minggu kemudian; jika tidak hati-hati, transaksi dekat batas Train/Val bisa “tercemar” oleh informasi pasca-batas.

## Apa itu “gap waktu / purge / embargo”

**Definisi:** jeda waktu yang sengaja dikosongkan di sekitar batas Train–Val/Test untuk memutus korelasi lintas-batas yang bisa bocor (fitur atau label).

**Intuisi:** jika label atau fitur bisa “menjalar” melintasi batas (karena konfirmasi/efek tertunda), maka observasi terlalu dekat dengan batas dikeluarkan dari kedua sisi.

## Apa itu forward-chaining

**Definisi:** skema split berurutan di waktu—selalu latih pada periode lebih awal, validasi pada periode setelahnya, lalu uji pada periode paling akhir.

**Bentuk umum:**

Train: [T0, T1) → Gap → Val: [T2, T3) → Gap → Test: [T4, T5)

Tidak pernah mengacak waktu; arah aliran data selalu “maju”.

## Cara menentukan besar “gap waktu” (resep praktis)

**Langkah 1 – Identifikasi label lag:** berapa lama rata-rata/maksimum jeda konfirmasi fraud (mis. chargeback)?

Contoh: median 21 hari, p95 = 45 hari.

**Langkah 2 – Identifikasi lookback fitur:** jendela agregasi terpanjang yang dipakai fitur (mis. 30 hari transaksi terakhir).

**Langkah 3 – Tetapkan gap konservatif:**

Rumus sederhana: gap = max(label_lag_aman, lookback_maks)

Contoh: label lag p95 = 45 hari, lookback fitur = 30 hari → gap = 45 hari.

Catatan: jika tersedia, gunakan kuantil tinggi (p95/p99) label lag untuk amankan ekor panjang.

## Contoh skema waktu (angka konkret)

Misal data Januari–Juni. Lookback = 30 hari; label lag p95 = 45 hari → gap = 45 hari.

- Train: 1 Jan – 31 Mar
- Gap: 1 Apr – 15 Mei (45 hari dikosongkan)
- Validation: 16 Mei – 31 Mei
- Gap: 1 Jun – 15 Jun (45 hari sebelum Test)
- Test: 16 Jun – 30 Jun

Selama feature engineering, setiap fitur berdasarkan timestampnya; agregasi 30 hari selalu dipotong di tanggal observasi (tidak pernah menengok ke depan).

## Implementasi teknis (ringkas, dapat ditempel ke metodologi)

**Penghitungan fitur:**

- Gunakan windowed aggregations berbasis waktu (rolling) yang dipotong di t (mis. sum(amount) for t-30d..t), bukan agregat keseluruhan dataset.

**Normalisasi/standarisasi:** fit hanya pada Train; simpan parameter scaler dan terapkan ke Val/Test.

**Pembentukan split:**

- Gunakan forward-chaining/TimeSeriesSplit; hapus baris jatuh di interval gap.
- Jika ada banyak transaksi per entitas (akun/perangkat), pertimbangkan StratifiedGroupKFold waktu + gap (agar entitas tidak overlap).

**Penetapan label:**

- Pastikan label “fraud” untuk observasi di Train tidak menggunakan informasi setelah akhir periode Train + gap.
- Jika label bergantung pada future window (mis. “fraud jika chargeback ≤ 60 hari”), purge minimal window label itu di batas split.

## Checklist audit (anti temporal leakage)

- Tidak ada normalisasi/encoding yang “melihat” data Val/Test (fit scaler/encoder di Train saja).
- Fitur agregat hanya pakai data ≤ t (tidak ada peek ahead).
- Diterapkan gap = max(label lag aman, lookback fitur) di setiap batas Train→Val dan Val→Test.
- Observasi dalam interval gap dikeluarkan dari kedua sisi.
- Untuk label berbasis masa depan (chargeback ≤ L hari), purge ≥ L hari di setiap batas.
- Semua keputusan di atas terdokumentasi (angka lag/lookback, panjang gap, ilustrasi kalender).

## Kesalahan umum (dan solusinya)

- **Kesalahan:** Mengacak baris seluruh periode → **Solusi:** gunakan forward-chaining.
- **Kesalahan:** Menghitung mean/target encoding pakai seluruh data → **Solusi:** OOF (out-of-fold) di Train; terapkan ke Val/Test.
- **Kesalahan:** Fitur jumlah 30 hari dihitung global (lintas split) → **Solusi:** rolling window per observasi, dibatasi pada tanggalnya.
- **Kesalahan:** Mengabaikan label lag → **Solusi:** ukur distribusi lag (p95/p99) dan set gap sesuai.

## Kalimat metodologi siap pakai (bahasa baku)

“Pembagian data dilakukan dengan skema forward-chaining berbasis waktu. Untuk mencegah temporal leakage, kami menerapkan interval gap di setiap batas Train–Validasi dan Validasi–Uji, dengan panjang gap ditetapkan sebagai max(kuantil tinggi keterlambatan konfirmasi label, jendela *lookback* fitur terpanjang). Observasi yang jatuh pada interval gap dikeluarkan dari pelatihan maupun evaluasi. Seluruh fitur agregat dihitung menggunakan jendela waktu yang dipotong pada timestamp observasi, dan seluruh transformasi (imputasi, standarisasi, encoding) di-fit hanya pada data latih dan kemudian diaplikasikan ke validasi/pengujian.”
