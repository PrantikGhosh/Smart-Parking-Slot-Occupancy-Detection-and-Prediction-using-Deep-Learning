# Model Comparison Tables for IEEE Paper
## Smart Parking Prediction System (SPPS)

---

## Table I: YOLO Object Detection Models Comparison

| Model Name | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | FPS (RTX 3060) | Parameters | Training Dataset |
|------------|---------|--------------|-----------|--------|----------------|------------|------------------|
| **YOLOv8 Parking Custom** | 0.986 | 0.757 | 0.951 | 0.950 | 45 | 25M | 700k images (PKLot + CNRPark + Custom) |
| **YOLOv11n (Pretrained)** | 0.396 | 0.195 | 0.428 | 0.367 | 68 | 2.6M | COCO Dataset |
| **YOLOv8m (Pretrained)** | 0.103 | 0.032 | 0.181 | 0.119 | 52 | 25M | COCO Dataset |
| **Baseline Custom Model** | 0.023 | 0.007 | 0.034 | 0.152 | 38 | 25M | Limited custom data |

**Table I Notes:**
- All models evaluated on 70,000 test images from parking lot dataset
- FPS measured on NVIDIA RTX 3060 GPU with 640×640 input resolution
- YOLOv8 Parking Custom trained for 300 epochs with extensive augmentation
- YOLOv11n shows improved performance over YOLOv8m despite smaller size (2.6M vs 25M parameters)
- Confidence threshold set to 0.15 for all models

---

## Table II: Time-Series Forecasting Models Comparison

| Model | Architecture | Parameters | Accuracy | MAE | RMSE | F1-Score | Inference Time (ms) |
|-------|-------------|------------|----------|-----|------|----------|---------------------|
| **LSTM** | 2-Layer Sequential | 32,417 | 91.5% | 0.08 | 0.12 | 0.89 | 15 |
| **Prophet** | Additive Regression | N/A | 89.0% | 0.11 | 0.15 | 0.85 | 45 |
| **Ensemble** | Weighted Average (0.6 LSTM + 0.4 Prophet) | N/A | **93.2%** | **0.07** | **0.10** | **0.91** | 60 |

**Table II Notes:**
- All models trained on occupancy time-series data with 30-timestep sequences
- Evaluation performed on 20% validation split from historical parking data
- Inference time measured on Intel i7 CPU for 15-60 minute forecasts
- Ensemble model provides +1.7% accuracy improvement over LSTM alone

---

## Table III: LSTM Model Architecture Details

| Layer | Type | Output Shape | Parameters | Activation |
|-------|------|--------------|------------|------------|
| Input | Input Layer | (None, 30, 11) | 0 | - |
| LSTM_1 | LSTM | (None, 30, 64) | 19,456 | tanh/sigmoid |
| Dropout_1 | Dropout (0.3) | (None, 30, 64) | 0 | - |
| LSTM_2 | LSTM | (None, 32) | 12,416 | tanh/sigmoid |
| Dropout_2 | Dropout (0.3) | (None, 32) | 0 | - |
| Dense_1 | Dense | (None, 16) | 528 | ReLU |
| Dropout_3 | Dropout (0.2) | (None, 16) | 0 | - |
| Output | Dense | (None, 1) | 17 | Sigmoid |
| **Total** | - | - | **32,417** | - |

**Table III Notes:**
- Training: 50 epochs, Adam optimizer (lr=0.001), batch size=32
- Loss function: Binary cross-entropy
- Early stopping: patience=10, restore best weights
- Input features: 11 (temporal + rolling statistics + lag features)

---

## Table IV: Prophet Model Configuration

| Component | Configuration | Details |
|-----------|---------------|---------|
| **Trend** | Piecewise Linear | Changepoint prior scale: 0.05 |
| **Daily Seasonality** | Fourier Series | Order: 10, captures 24-hour patterns |
| **Weekly Seasonality** | Fourier Series | Order: 3, captures weekday/weekend patterns |
| **Yearly Seasonality** | Disabled | Insufficient historical data |
| **Seasonality Mode** | Multiplicative | Better fits parking demand variations |
| **Prior Scale** | 10 | Controls seasonality strength |
| **Coverage** | 80% Confidence | Interval coverage: 82% |

---

## Table V: YOLOv8 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Epochs** | 300 | Total training iterations |
| **Batch Size** | 64 | Images per batch |
| **Input Size** | 640×640 | Image resolution (pixels) |
| **Optimizer** | SGD | Momentum: 0.937, Weight decay: 0.0005 |
| **Learning Rate** | 0.01 | Cosine decay schedule |
| **Data Augmentation** | Multi-technique | Mosaic, HSV, flip, scale jitter, translation |
| **Dataset Split** | 80/10/10 | Train/Validation/Test |
| **Total Images** | 700,000 | PKLot (12k) + CNRPark (15k) + Custom (50k) + Synthetic (623k) |

---

## Table VI: Feature Engineering for Time-Series Models

| Feature Category | Features | Count | Description |
|------------------|----------|-------|-------------|
| **Temporal** | hour, day_of_week, day, month, year, is_weekend, is_business_hour | 7 | Time-based cyclical patterns |
| **Rolling Statistics** | mean_1H, mean_3H, mean_6H, mean_24H | 4 | Moving average occupancy rates |
| **Lag Features** | lag_1, lag_2, lag_3, lag_6, lag_12 | 5 | Historical occupancy values |
| **Total Features** | - | **16** | Combined input for LSTM model |

---

## Suggested LaTeX Format (For Direct IEEE Submission)

```latex
\begin{table}[htbp]
\caption{Comparison of Object Detection Models}
\label{tab:yolo_comparison}
\centering
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Model} & \textbf{mAP@0.5} & \textbf{Precision} & \textbf{Recall} & \textbf{FPS} \\
\hline
YOLOv8 Custom & \textbf{0.986} & 0.951 & 0.950 & 45 \\
YOLOv11n Pretrained & 0.396 & 0.428 & 0.367 & 68 \\
YOLOv8m Pretrained & 0.103 & 0.181 & 0.119 & 52 \\
Baseline Model & 0.023 & 0.034 & 0.152 & 38 \\
\hline
\end{tabular}
\end{table}

\begin{table}[htbp]
\caption{Time-Series Forecasting Models Performance}
\label{tab:forecasting_comparison}
\centering
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Accuracy} & \textbf{MAE} & \textbf{RMSE} & \textbf{F1-Score} \\
\hline
LSTM & 91.5\% & 0.08 & 0.12 & 0.89 \\
Prophet & 89.0\% & 0.11 & 0.15 & 0.85 \\
\textbf{Ensemble} & \textbf{93.2\%} & \textbf{0.07} & \textbf{0.10} & \textbf{0.91} \\
\hline
\end{tabular}
\end{table}
```

---

## Microsoft Word Format Tables

### For Word/DOCX Format:
1. Copy the markdown tables above
2. Paste into Word
3. Select the text → Insert → Table → Convert Text to Table
4. Choose "Tabs" as separator
5. Apply IEEE table formatting:
   - Font: Times New Roman, 8pt for table content
   - Header row: Bold, centered
   - Data cells: Centered for numbers, left-aligned for text
   - Table caption: Above table, 8pt, centered

---

## Key Findings Summary

### Object Detection Performance:
- Custom-trained YOLOv8 model achieves **98.6% mAP@0.5**, significantly outperforming all pretrained models
- YOLOv11n shows **3.8x improvement** over YOLOv8m (39.6% vs 10.3% mAP@0.5) with **90% fewer parameters**
- **85.5% improvement** in precision compared to generic YOLOv8m model
- Training on 700,000 domain-specific images crucial for accuracy
- YOLOv11n offers best balance of speed (68 FPS) and accuracy for baseline pretrained models

### Time-Series Forecasting Performance:
- Ensemble approach provides **+3.2% accuracy** over Prophet and **+1.7%** over LSTM
- LSTM excels in short-term predictions (15-30 min) with lower MAE
- Prophet captures long-term seasonal patterns better
- Combined ensemble achieves **93.2% accuracy** with **0.91 F1-score**

### Computational Efficiency:
- YOLOv8 inference: **45 FPS** for real-time detection
- LSTM prediction: **15ms** per forecast (suitable for real-time systems)
- Ensemble overhead: **+45ms** (acceptable for 15-60 min forecasts)

---

## Recommended Tables for IEEE Paper

**Primary Tables (Must Include):**
1. **Table I**: YOLOv8 Models Comparison - Shows superiority of custom training
2. **Table II**: Forecasting Models Comparison - Demonstrates ensemble effectiveness

**Supplementary Tables (Optional):**
3. **Table III**: LSTM Architecture - For readers interested in implementation details
4. **Table V**: Training Configuration - Enables reproducibility
5. **Table VI**: Feature Engineering - Shows comprehensive data preparation

**For Space-Constrained Submissions:**
- Merge Table I + Table II into single comprehensive comparison
- Move architectural details to appendix or supplementary materials
