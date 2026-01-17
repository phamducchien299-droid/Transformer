# Transformer Time Series Forecasting

A **Transformer-based deep learning model** for time series forecasting, designed to predict the Oil Temperature (OT) from the ETTh1 electricity transformer dataset.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `transformer_forecasting.py` | Main Python script with complete ML pipeline |
| `ETTh1.csv` | Dataset (~17,421 hourly samples) |
| `requirements.txt` | Python dependencies |
| `best_model.pth` | Saved best model weights |
| `results_*.png` | Training visualization outputs |
| `sample_data.csv` | Smaller sample dataset |

---

## 🏗️ Architecture

The pipeline consists of 7 main components:

1. **`TimeSeriesDataset`** - Custom PyTorch Dataset for sliding window sequences
2. **`PositionalEncoding`** - Sinusoidal position encodings for transformer
3. **`TransformerForecaster`** - Main model using `nn.TransformerEncoder`
4. **`train_model()`** - Training loop with validation & LR scheduling
5. **`evaluate_model()`** - Metrics: MSE, MAE, RMSE, MAPE
6. **`run_forecasting_pipeline()`** - End-to-end pipeline for multiple horizons
7. **Main block** - Runs predictions for 96h, 192h, 336h, 720h

---

## 📊 Dataset: ETTh1

| Property | Value |
|----------|-------|
| **Rows** | 17,421 hourly samples (July 2016 - July 2018) |
| **Features** | `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT` |
| **Target** | `OT` (Oil Temperature) |

---

## 🔧 Model Configuration

| Parameter | Value |
|-----------|-------|
| Input dim | 7 features |
| d_model | 128 |
| Attention heads | 8 |
| Encoder layers | 3 |
| Feedforward dim | 512 |
| Dropout | 0.1 |
| Input sequence | 96 time steps |
| Prediction lengths | 96h, 192h, 336h, 720h |

---

## 📦 Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
torch>=1.10.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run forecasting pipeline
python transformer_forecasting.py
```

**Output:**
- Model weights saved to `best_model.pth`
- Results saved to `forecasting_results.json`
- Visualizations saved as `results_*.png`
