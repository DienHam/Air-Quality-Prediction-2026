# Phân tích mô hình và quy trình train — PM2.5 t+1 Hà Nội

Tài liệu này mô tả **bài toán**, **từng mô hình** (baseline → cải tiến), **quy trình train/đánh giá**, và **các chỗ dễ sai** khi huấn luyện trên dữ liệu đã crawl.

Liên quan dữ liệu:

| File | Vai trò |
| --- | --- |
| `data/interim/hourly_hanoi.parquet` | Chuỗi giờ OpenAQ + ERA5 + CAMS (nhãn gốc trạm) |
| `data/interim/hourly_train_six.parquet` | 6 trạm, nhãn `pm25_train` đã bù OpenWeather/CAMS |
| `data/interim/stations.csv` / `stations_train_six.csv` | Metadata + độ phủ |

---

## 1. Bài toán

### 1.1 Mục tiêu

Dự báo nồng độ **PM2.5 tại giờ tiếp theo (t+1)** tại từng trạm quan trắc / điểm theo dõi ở Hà Nội, đơn vị **µg/m³**.

\[
\hat{y}_{s,t+1} = f\!\left(x_{s,t}\right)
\]

- \(s\): trạm (`location_id`)
- \(t\): giờ địa phương `Asia/Ho_Chi_Minh`
- \(y_{s,t+1}\): PM2.5 thực tại giờ sau (nhãn)
- \(x_{s,t}\): đặc trưng **chỉ dùng thông tin có tại hoặc trước giờ \(t\)** (không nhìn tương lai)

Kết quả dùng cảnh báo ngắn hạn, **không thay** trạm quan trắc.

### 1.2 Metric

| Metric | Công thức ý tưởng | Ghi chú |
| --- | --- | --- |
| **MAE** | trung bình \|ŷ − y\| | Chính, dễ diễn giải (µg/m³) |
| **RMSE** | √ trung bình (ŷ − y)² | Phạt lệch lớn (đỉnh ô nhiễm) |

Báo cáo thêm (khuyến nghị):

- MAE/RMSE **theo trạm**
- MAE **theo giờ trong ngày** (0–23)
- MAE **mùa khô (11–4) vs mùa mưa**
- So với baseline persistence: **skill** = 1 − MAE_model / MAE_B0

### 1.3 Hai lớp nhãn — không được trộn khi chấm điểm chính

| Lớp | Cột | Nguồn | Dùng để |
| --- | --- | --- | --- |
| **A – Trạm (chính)** | `pm25` / `pm25_openaq` | OpenAQ S3 | Metric chính, báo cáo đồ án |
| **B – Mixed (phụ)** | `pm25_train` | OpenAQ → OpenWeather → CAMS | Ablation, thử 6 trạm đủ phủ |

**Quy tắc:** mọi bảng kết quả “chính thức” chỉ tính trên giờ có `pm25_source == "openaq"` (hoặc tương đương: nhãn lấy từ OpenAQ). Tập mixed chỉ nêu rõ là thí nghiệm phụ.

---

## 2. Dữ liệu đưa vào train

### 2.1 Cửa sổ và độ phân giải

- Độ phân giải: **1 giờ / trạm** (mẫu 5 phút đã trung bình trong giờ).
- Cửa sổ hiện tại (recrawl): khoảng **2025-10-01 → 2026-11-25**.
- Trạm OpenAQ đủ phủ thuần (≥60% PM2.5): **Nhân Chính, ĐHBK, OceanPark**.
- Ba trạm còn lại (Minh Khai, Lưu Quang Vũ, Nguyễn Văn Cừ) thiếu nhiều trên S3; file six đã bù model.

### 2.2 Đặc trưng có sẵn (sau crawl)

**Nhãn / mục tiêu**

- `pm25_next` = PM2.5 tại t+1 (tính bằng `shift(-1)` theo từng trạm)

**Autoregressive / rolling (tính riêng từng trạm, sau `asfreq("h")`)**

- `pm25`, `pm25_lag1`, `pm25_lag24`, `pm25_lag168`
- `pm25_roll_3h`, `pm25_roll_24h`, `pm25_roll_168h`, `pm25_roll_24h_std`

**Thời tiết ERA5 (Open-Meteo Archive)**

- `temperature_2m`, `relative_humidity_2m`, `surface_pressure`
- `wind_speed_10m`, `wind_u`, `wind_v`, `precipitation`, `boundary_layer_height`

**Nền CAMS Global (một điểm Hà Nội, lặp mọi trạm)**

- `cams_pm25`, `cams_pm10`, `cams_no2`, `cams_o3`, `cams_dust`, `cams_aod`

**Lịch**

- `hour_of_day`, `day_of_week`, `month`, `is_dry_season`
- `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`

### 2.3 Bộ feature đã chốt theo giai

| Giai đoạn | Feature | Ghi chú |
| --- | --- | --- |
| **V1 (tối thiểu)** | `pm25_lag1`, `pm25_lag24`, `wind_speed_10m` | Khớp plan crawl / README |
| **V2 (bổ sung)** | + lịch sin/cos, ERA5, rolling, CAMS (trừ `cams_pm10` nếu trùng `cams_pm25`) | Chỉ thêm khi đủ phủ và \|corr\| &lt; 0.95 với feature đã giữ |
| **Không đưa vào nhãn** | CAMS, OpenWeather, ERA5 | Chỉ được là **đặc trưng** (hoặc nhãn phụ nếu thí nghiệm mixed) |

---

## 3. Phân tích chi tiết từng mô hình

### 3.1 B0 — Persistence (baseline cứng)

**Ý tưởng:** giờ sau ≈ giờ này.

\[
\hat{y}_{t+1} = \mathrm{PM2.5}_t
\]

| Mục | Chi tiết |
| --- | --- |
| Input | `pm25` tại t |
| Ưu | Không train; phản ánh tự tương quan rất mạnh của PM2.5 |
| Nhược | Trễ 1 giờ khi có front gió / mưa / đỉnh giao thông |
| Khi nào thắng | Chuỗi ổn định, ít biến động giờ-ke-giờ |
| Cách báo cáo | Luôn in MAE/RMSE B0 trên **cùng tập test** với mọi model khác |

**Chú ý:** đây là mốc “khó đánh bại”. Model phức tạp mà không thắng B0 trên test thì chưa đủ thuyết phục.

---

### 3.2 B1 — Seasonal persistence (cùng giờ hôm qua)

\[
\hat{y}_{t+1} = \mathrm{PM2.5}_{t-23}
\]

(tương đương dùng giá trị cách 24 giờ so với nhãn t+1, tức PM2.5 tại cùng giờ ngày trước.)

| Mục | Chi tiết |
| --- | --- |
| Input | Chuỗi PM2.5 đủ `lag24` |
| Ưu | Bắt chu kỳ ngày (ban ngày thấp hơn / đêm cao hơn tùy mùa) |
| Nhược | Không phản ứng thời tiết ngày hiện tại |
| So với B0 | Thường kém B0 trên t+1, nhưng hữu ích như baseline thứ hai |

---

### 3.3 B2 — Moving average

\[
\hat{y}_{t+1} = \mathrm{mean}(\mathrm{PM2.5}_{t-k+1},\ldots,\mathrm{PM2.5}_t)
\]

Với k = 3 hoặc 24 (`pm25_roll_3h` / `pm25_roll_24h`).

| Mục | Chi tiết |
| --- | --- |
| Ưu | Làm mượt nhiễu cảm biến |
| Nhược | Chậm hơn B0 khi có đột biến |
| Vai trò | Baseline phụ; ít khi là model cuối |

---

### 3.4 V1 — Hồi quy tuyến tính (mô hình cải tiến tối thiểu)

\[
\hat{y}_{t+1} = \beta_0 + \beta_1\,\mathrm{pm25\_lag1} + \beta_2\,\mathrm{pm25\_lag24} + \beta_3\,\mathrm{wind\_speed\_10m}
\]

| Mục | Chi tiết |
| --- | --- |
| Thuật toán | `LinearRegression` hoặc `Ridge` (α nhỏ) |
| Feature | Đúng bộ V1 |
| Ưu | Giải thích được (gió mạnh ↔ PM giảm, …); ít overfit |
| Nhược | Không bắt tương tác phi tuyến (gió × giờ đêm, mưa × BLH) |
| Kỳ vọng | Thắng B0 nhẹ–vừa trên MAE nếu gió và chu kỳ ngày có tín hiệu |

**Chuẩn hóa:** với Ridge/LR, có thể `StandardScaler` chỉ fit trên train. Không bắt buộc với tree models.

---

### 3.5 V2 — Gradient boosting (mô hình chính đề xuất)

**Ứng viên:** LightGBM (ưu tiên), XGBoost, CatBoost.

| Mục | Chi tiết |
| --- | --- |
| Target | `pm25_next` |
| Feature V2 | V1 + lịch + ERA5 + rolling + CAMS (+ `location_id` nếu train chung nhiều trạm) |
| Loss | MAE (Huber / quantile tùy thư viện) hoặc MSE rồi báo cả MAE |
| Ưu | Phi tuyến, tương tác feature, mạnh trên tabular giờ |
| Nhược | Dễ “học” leakage nếu có cột tương lai; cần early stopping trên validation thời gian |

**Siêu tham số gợi ý (điểm xuất phát):**

- `n_estimators`: 300–2000 + early stopping
- `learning_rate`: 0.03–0.1
- `num_leaves` / `max_depth`: vừa phải (tránh cây quá sâu trên ~2–6×10⁴ dòng)
- `min_child_samples`: ≥ 20–50
- `subsample`, `colsample_bytree`: 0.7–0.9

**Giải thích:** SHAP hoặc `feature_importances_` — kỳ vọng `pm25_lag1` chiếm tỉ trọng lớn; lịch và gió/CAMS bổ sung.

---

### 3.6 V3 — Multi-station / tín hiệu không gian

**Ý tưởng:** PM2.5 trạm gần thường đồng biến; dùng thông tin hàng xóm tại **cùng giờ t**.

Ví dụ feature thêm:

- `pm25` của 1–2 trạm gần nhất (cùng t)
- Khoảng cách (km) hoặc trọng số 1/distance
- Hoặc một model chung mọi trạm + categorical `location_id`

| Mục | Chi tiết |
| --- | --- |
| Ưu | Bù khi một trạm thiếu / nhiễu; bắt plume đô thị |
| Nhược | Phụ thuộc đồng bộ giờ giữa trạm; khoảng cách lớn (OceanPark) yếu hơn |
| Điều kiện | Chỉ lấy hàng xóm **đã quan sát tại t**, không dùng y_{t+1} của hàng xóm |

---

### 3.7 V4 — Mô hình chuỗi (LSTM / TFT / N-BEATS) — tùy chọn

| Mục | Chi tiết |
| --- | --- |
| Input | Cửa sổ 24–168 giờ PM2.5 (+ optional weather) theo trạm |
| Khi nào làm | Sau khi V2 đã solid; horizon vẫn chỉ 1h nên lợi thế thường nhỏ |
| Rủi ro | Cần chuỗi đủ liên tục → ưu tiên 3 trạm OpenAQ, tránh đoạn bù OWM dài |
| Kết luận thực dụng | **Không ưu tiên** cho MVP đồ án nếu V2 đã thắng B0 rõ |

---

### 3.8 V5 — Phân tầng / robust

- Train riêng **mùa khô / mùa mưa**, hoặc thêm `is_dry_season` (đã có trong V2).
- Sample weight: giờ OpenAQ weight = 1; giờ chỉ OWM weight ≪ 1 (nếu vẫn dùng mixed).
- Clip nhãn/outlier cực đoan (ví dụ Minh Khai từng có spike rất cao) — ghi rõ ngưỡng nếu dùng.

---

## 4. Quy trình train (end-to-end)

```
[1] Chọn nguồn nhãn
      ├─ Chính: hourly_hanoi.parquet, chỉ dòng OpenAQ đủ lag
      └─ Phụ:   hourly_train_six.parquet (ablation)
[2] Lọc dòng hợp lệ
      └─ Đủ pm25_next + feature; không nội suy PM2.5 để “lấp” nhãn chính
[3] Chia thời gian (theo từng trạm hoặc global time)
      ├─ Train: phần đầu
      ├─ Valid: đoạn giữa (early stopping / chọn model)
      └─ Test:  tháng cuối (hold-out)
[4] Fit baseline B0 (và B1) trên test cùng mask
[5] Fit V1 → V2 (early stopping trên valid)
[6] Đánh giá MAE/RMSE + skill vs B0
[7] Phân tích lỗi theo trạm / giờ / mùa
[8] (Tuỳ) V3 không gian, ablation mixed labels
```

### 4.1 Bước 1 — Chọn tập và lọc

**Tập chính (khuyến nghị báo cáo):**

1. Lấy 3 trạm `in_train` từ OpenAQ thuần (Nhân Chính, ĐHBK, OceanPark), **hoặc** mọi giờ có `pm25` OpenAQ trên 6 trạm.
2. Giữ dòng có đủ: `pm25_next`, `pm25_lag1`, `pm25_lag24`, `wind_speed_10m` (và feature V2 nếu dùng).
3. **Không** điền nhãn bằng nội suy tuyến tính giữa hai giờ OpenAQ để tạo thêm mẫu train chính.

**Tập phụ (six mixed):**

- Dùng `pm25_train` / `pm25_next` đã tính trên `pm25_train`.
- Luôn tách metric: toàn mixed vs chỉ subset OpenAQ.

### 4.2 Bước 2 — Chia train / validation / test

**Nguyên tắc:** chia theo **thời gian**, không `train_test_split` ngẫu nhiên.

Gợi ý cho cửa sổ ~14 tháng:

| Tập | Tỉ lệ thời gian (ví dụ) | Mục đích |
| --- | --- | --- |
| Train | ~70% giờ đầu | Fit |
| Valid | ~15% tiếp | Early stopping, chọn siêu tham số |
| Test | ~15% cuối (hoặc **2–3 tháng cuối**) | Báo cáo duy nhất |

Làm **theo từng trạm** cùng mốc cắt ngày (cùng `split_date`) để so sánh công bằng giữa trạm.

Ví dụ: nếu hết dữ liệu 2026-11-22, có thể cắt test = từ 2026-09-01 hoặc 2026-10-01 trở đi — **chọn một lần và giữ cố định**.

### 4.3 Bước 3 — Train từng mô hình

1. **B0:** trên test, `ŷ = pm25` (tại t); không fit.
2. **V1:** fit Ridge/LR trên train; đánh giá valid → test.
3. **V2:** fit boosting trên train; early stopping theo MAE valid; khóa `best_iteration`; đánh giá test **một lần**.
4. Lưu: model file, danh sách feature, ngày cắt split, git commit / hash dữ liệu.

### 4.4 Bước 4 — Đánh giá và bảng so sánh

Mẫu bảng báo cáo:

| Model | MAE test | RMSE test | Skill vs B0 | Ghi chú |
| --- | ---: | ---: | ---: | --- |
| B0 Persistence | | | 0 | |
| B1 Seasonal | | | | |
| V1 Linear | | | | |
| V2 LightGBM | | | | |

Thêm bảng theo `location_id`.

### 4.5 Bước 5 — Kiểm tra leakage (bắt buộc trước khi công bố số)

Checklist:

- [ ] Không có cột của giờ t+1 trong X (trừ target tách riêng).
- [ ] `cams_*`, ERA5 dùng giá trị **tại t**, không phải forecast t+1 trừ khi API ghi rõ và bạn chấp nhận giả định.
- [ ] Rolling/lag chỉ dùng quá khứ trên cùng trạm.
- [ ] StandardScaler / encoding **fit trên train**, transform valid/test.
- [ ] Early stopping chỉ nhìn valid; test không dùng để chọn epoch/siêu tham số lần cuối (tránh “test overfitting”).

---

## 5. Chú ý quan trọng khi train

### 5.1 Rò rỉ thời gian (time leakage)

| Sai | Đúng |
| --- | --- |
| Xáo hàng rồi chia 80/20 | Cắt theo ngày/tháng |
| Dùng `pm25_next` làm feature | Chỉ làm y |
| Normalize trên cả train+test | Fit scaler trên train |
| Chọn model theo điểm test | Chọn theo valid, test một lần |

### 5.2 Nhãn lẫn model (OpenWeather / CAMS)

- OpenWeather và CAMS là **mô hình / tái phân tích**, không phải cảm biến trạm.
- Train trên `pm25_train` mixed có thể làm MAE “đẹp” giả tạo nếu test cũng chứa nhiều OWM.
- **Bắt buộc** nêu tỉ lệ nguồn nhãn trên train và trên test.

### 5.3 Thiếu giờ và lag

- `shift` qua đoạn NaN tạo mẫu xấu nếu fill tùy tiện.
- Quy tắc an toàn: **drop** mọi dòng thiếu bất kỳ feature/lag/nhãn đang dùng.
- Không nội suy PM2.5 để tạo nhãn chính.

### 5.4 Khác biệt giữa trạm

- OceanPark: cảm biến AirGradient (low-cost), giấy phép CC BY; hành vi có thể khác mạng chuẩn.
- Nhân Chính / ĐHBK: chuỗi dày, đo ~5 phút rồi aggregate giờ.
- Minh Khai / Lưu Quang Vũ / Nguyễn Văn Cừ: nhiều gap; không đánh đồng chất lượng nhãn với 3 trạm train thuần.

Nên báo MAE **từng trạm**, không chỉ số gộp.

### 5.5 ERA5 cùng giờ

ERA5 trong archive là tái phân tích cùng giờ t — **không** phải bản tin phát hành trước t. Với bài tập t+1 học thuật chấp nhận được; trong báo cáo **phải nói rõ** giới hạn này nếu triển khai realtime cần weather forecast thay vì reanalysis.

### 5.6 Overfitting và độ khó baseline

- PM2.5 lag-1 thường giải thích phần lớn phương sai → tree sâu dễ memorization.
- Luôn so B0; nếu skill &lt; 5–10% trên test, kiểm lại feature/leakage trước khi thêm model phức tạp.

### 5.7 Đơn vị và outlier

- Mọi PM về **µg/m³**.
- Kiểm tra min/max theo trạm; cân nhắc winsorize chỉ trên **train** nếu có spike cảm biến (ghi rõ percentile).

### 5.8 Tái lập (reproducibility)

Lưu cùng lúc:

1. File parquet + checksum hoặc ngày export  
2. `split_date`, danh sách `location_id`  
3. Seed, phiên bản thư viện (`lightgbm`, `sklearn`, `pandas`)  
4. Script train trong `scripts/` và commit hash  

---

## 6. Đề xuất cấu trúc thư mục khi code train

```
scripts/
  train_baselines.py      # B0, B1, B2
  train_v1_linear.py
  train_v2_boosting.py
  eval_report.py          # MAE/RMSE theo trạm, skill vs B0
models/                   # gitignore hoặc chỉ lưu model nhỏ
reports/models/
  metrics_test.csv
  metrics_by_station.csv
  feature_importance.png
```

Output tối thiểu của một lần chạy:

- `reports/models/metrics_test.csv`
- Hình: time series ŷ vs y trên 1–2 tuần test của một trạm
- Bảng skill vs B0

---

## 7. Thứ tự triển khai khuyến nghị

1. **B0 + B1** trên 3 trạm OpenAQ, test tháng cuối.  
2. **V1** cùng split — xác nhận thắng B0.  
3. **V2 LightGBM** + early stopping — model chính.  
4. Phân tích lỗi theo giờ/mùa/trạm.  
5. (Tuỳ) **V3** hàng xóm; ablation trên `hourly_train_six`.  
6. (Tuỳ) V4 deep sequence chỉ nếu còn mục tiêu nghiên cứu.

---

## 8. Tóm tắt một trang

| Hạng mục | Chọn |
| --- | --- |
| Bài toán | PM2.5 t+1, theo trạm, theo giờ |
| Metric chính | MAE, RMSE trên nhãn OpenAQ |
| Baseline | Persistence (B0) |
| Model chính | LightGBM (V2) |
| Model tối thiểu cải tiến | Linear V1 (lag1, lag24, gió) |
| Chia dữ liệu | Theo thời gian; test = đoạn cuối |
| Cấm | Xáo hàng; dùng tương lai; chấm mixed như OpenAQ |
| File nhãn chính | `hourly_hanoi.parquet` |
| File thí nghiệm 6 trạm | `hourly_train_six.parquet` (`pm25_train`) |

---

*Phiên bản tài liệu bám dữ liệu crawl/recrawl Hà Nội và file train six tại thời điểm soạn. Khi đổi cửa sổ thời gian hoặc danh sách trạm, cập nhật lại mục split và bảng độ phủ.*
