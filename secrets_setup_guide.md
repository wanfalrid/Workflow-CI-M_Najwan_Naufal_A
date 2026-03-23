# Panduan Setup GitHub Secrets

**Repository:** Workflow-CI  
**Student:** M_Najwan_Naufal_A

---

## Cara Set GitHub Secrets

1. Buka repository **Workflow-CI** di GitHub
2. Klik **Settings** (tab paling kanan)
3. Di sidebar kiri, klik **Secrets and variables** → **Actions**
4. Klik **New repository secret**
5. Isi **Name** dan **Secret**, lalu klik **Add secret**

---

## Daftar Secrets yang Dibutuhkan

### 1. `DOCKERHUB_USERNAME`
- **Apa:** Username Docker Hub kamu
- **Cara dapat:**
  1. Buka https://hub.docker.com
  2. Login / Sign up
  3. Username terlihat di kanan atas
- **Contoh value:** `wanfalrid`

### 2. `DOCKERHUB_TOKEN`
- **Apa:** Access Token Docker Hub (bukan password!)
- **Cara dapat:**
  1. Login ke https://hub.docker.com
  2. Klik avatar → **Account Settings**
  3. Klik **Security** → **Personal access tokens**
  4. Klik **Generate New Token**
  5. Beri nama (misal: `github-actions`)
  6. Pilih permission: **Read & Write**
  7. Copy tokennya (hanya muncul sekali!)
- **Contoh value:** `dckr_pat_xxxxx...xxxxx`

### 3. `DAGSHUB_USERNAME`
- **Apa:** Username DagsHub kamu
- **Cara dapat:**
  1. Login ke https://dagshub.com
  2. Username terlihat di profil
- **Contoh value:** `wanfalrid`

### 4. `DAGSHUB_TOKEN`
- **Apa:** Access Token DagsHub
- **Cara dapat:**
  1. Login ke https://dagshub.com
  2. Klik avatar → **Settings**
  3. Klik **Tokens** di sidebar
  4. Klik **Generate New Token**
  5. Copy tokennya
- **Contoh value:** `a1b2c3d4e5f6...`

### 5. `MLFLOW_TRACKING_URI`
- **Apa:** URL MLflow tracking server di DagsHub
- **Format:** `https://dagshub.com/<username>/<repo>.mlflow`
- **Contoh value:** `https://dagshub.com/wanfalrid/Eksperimen_SML_M_Najwan_Naufal_A.mlflow`

---

## Checklist

Setelah menambahkan semua secrets, pastikan halaman Secrets menunjukkan:

```
Repository secrets (5)
├── DAGSHUB_TOKEN          Updated just now
├── DAGSHUB_USERNAME       Updated just now
├── DOCKERHUB_TOKEN        Updated just now
├── DOCKERHUB_USERNAME     Updated just now
└── MLFLOW_TRACKING_URI    Updated just now
```

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Docker push gagal `denied` | Cek DOCKERHUB_TOKEN punya permission Read & Write |
| MLflow tracking error 401 | Cek DAGSHUB_TOKEN masih valid (belum expired) |
| `secret not found` di log | Nama secret CASE SENSITIVE — harus persis |
| CI masih pakai local tracking | Pastikan MLFLOW_TRACKING_URI terisi dengan benar |
