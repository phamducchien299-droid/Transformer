import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os

# Kiểm tra GPU
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))

# Cấu hình GPU để tận dụng tối đa
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)

# ============================================
# 1. POSITIONAL ENCODING
# ============================================
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Tạo positional encoding matrix
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[tf.newaxis, :seq_len, :]

# ============================================
# 2. TRANSFORMER MODEL
# ============================================
def build_transformer_model(input_dim=7, seq_len=96, pred_len=96, 
                           d_model=128, num_heads=8, num_layers=3,
                           dff=512, dropout_rate=0.1, output_dim=1):
    """
    Xây dựng mô hình Transformer cho dự báo chuỗi thời gian
    
    Args:
        input_dim: Số lượng features đầu vào (7 cho ETTh1)
        seq_len: Độ dài sequence đầu vào (96)
        pred_len: Độ dài dự báo (96, 192, 336, 720)
        d_model: Dimension của model (128)
        num_heads: Số attention heads (8)
        num_layers: Số encoder layers (3)
        dff: Dimension của feedforward network (512)
        dropout_rate: Tỷ lệ dropout (0.1)
        output_dim: Số features đầu ra (1 - chỉ OT)
    """
    # Input layer
    inputs = layers.Input(shape=(seq_len, input_dim), name='input')
    
    # Input projection
    x = layers.Dense(d_model, name='input_projection')(inputs)
    
    # Positional encoding
    x = PositionalEncoding(d_model)(x)
    
    # Transformer Encoder Layers
    for i in range(num_layers):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f'mha_{i}'
        )(x, x)
        
        # Add & Norm
        x = layers.LayerNormalization(epsilon=1e-6, name=f'ln1_{i}')(x + attn_output)
        
        # Feedforward
        ffn = keras.Sequential([
            layers.Dense(dff, activation='relu', name=f'ffn1_{i}'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model, name=f'ffn2_{i}')
        ], name=f'ffn_{i}')
        
        ffn_output = ffn(x)
        
        # Add & Norm
        x = layers.LayerNormalization(epsilon=1e-6, name=f'ln2_{i}')(x + ffn_output)
    
    # Lấy token cuối cùng
    x = x[:, -1, :]
    
    # Output projection
    x = layers.Dense(pred_len * output_dim, name='output_projection')(x)
    
    # Reshape to (batch, pred_len, output_dim)
    outputs = layers.Reshape((pred_len, output_dim), name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='transformer_forecaster')
    
    return model

# ============================================
# 3. DATA PREPARATION
# ============================================
def create_dataset(data, seq_len=96, pred_len=96, target_col=-1, batch_size=32, shuffle=True):
    """
    Tạo TensorFlow Dataset
    """
    n_samples = len(data) - seq_len - pred_len + 1
    
    X = np.zeros((n_samples, seq_len, data.shape[1]))
    y = np.zeros((n_samples, pred_len, 1))
    
    for i in range(n_samples):
        X[i] = data[i:i + seq_len]
        y[i] = data[i + seq_len:i + seq_len + pred_len, target_col:target_col+1]
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

# ============================================
# 4. TRAINING CALLBACKS
# ============================================
def get_callbacks(model_save_path, patience=10):
    """
    Tạo callbacks cho training
    """
    callbacks = [
        # Lưu model tốt nhất
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # Giảm learning rate khi plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=f'./logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    return callbacks

# ============================================
# 5. EVALUATION FUNCTION
# ============================================
def evaluate_model(model, test_dataset, scaler, pred_len):
    """
    Đánh giá model
    """
    predictions = []
    actuals = []
    
    for batch_x, batch_y in test_dataset:
        pred = model.predict(batch_x, verbose=0)
        predictions.append(pred)
        actuals.append(batch_y.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # Inverse transform
    pred_shape = predictions.shape
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(pred_shape)
    actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(pred_shape)
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'predictions': predictions,
        'actuals': actuals
    }

# ============================================
# 6. MAIN PIPELINE
# ============================================
def run_forecasting_pipeline(data_path, pred_lengths=[96, 192, 336, 720], 
                            target_col='OT', epochs=50, batch_size=32):
    """
    Pipeline chính cho dự báo
    """
    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Feature columns
    feature_cols = [col for col in df.columns if col.lower() not in ['date', 'datetime', 'time']]
    print(f"\nFeature columns: {feature_cols}")
    
    data = df[feature_cols].values
    
    # Target column index
    target_idx = feature_cols.index(target_col) if target_col in feature_cols else -1
    print(f"Target column: {target_col} (index: {target_idx})")
    
    # Split data
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_data = data[:train_end]
    val_data = data[:val_end]
    test_data = data[val_end:]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data) - len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Standardization
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)
    
    # Target scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df[target_col].values[:train_end].reshape(-1, 1))
    
    # Results dictionary
    all_results = {}
    
    # Train for each prediction length
    for pred_len in pred_lengths:
        print(f"\n{'='*60}")
        print(f"Training for prediction length: {pred_len}h")
        print(f"{'='*60}")
        
        seq_len = 96
        
        # Create datasets
        train_dataset = create_dataset(train_data, seq_len, pred_len, target_idx, batch_size, shuffle=True)
        val_dataset = create_dataset(val_data, seq_len, pred_len, target_idx, batch_size, shuffle=False)
        test_dataset = create_dataset(test_data, seq_len, pred_len, target_idx, batch_size, shuffle=False)
        
        # Build model
        model = build_transformer_model(
            input_dim=len(feature_cols),
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=128,
            num_heads=8,
            num_layers=3,
            dff=512,
            dropout_rate=0.1,
            output_dim=1
        )
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"\nModel summary:")
        model.summary()
        
        # Model save path
        model_save_path = f'best_model_{pred_len}h.h5'
        
        # Train
        print(f"\nStarting training...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=get_callbacks(model_save_path, patience=10),
            verbose=1
        )
        
        # Load best model
        model.load_weights(model_save_path)
        
        # Evaluate
        print(f"\nEvaluating model...")
        results = evaluate_model(model, test_dataset, target_scaler, pred_len)
        
        print(f"\nResults for {pred_len}h prediction:")
        print(f"  MSE: {results['MSE']:.4f}")
        print(f"  MAE: {results['MAE']:.4f}")
        print(f"  RMSE: {results['RMSE']:.4f}")
        print(f"  MAPE: {results['MAPE']:.2f}%")
        
        # Save results
        all_results[f'{pred_len}h'] = {
            'MSE': results['MSE'],
            'MAE': results['MAE'],
            'RMSE': results['RMSE'],
            'MAPE': results['MAPE'],
            'train_loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
        
        # Plot results
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Training curves
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Curves - {pred_len}h')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Sample prediction
        plt.subplot(1, 3, 2)
        plt.plot(results['actuals'][0], label='Actual', marker='o', markersize=3)
        plt.plot(results['predictions'][0], label='Predicted', marker='x', markersize=3)
        plt.xlabel('Time Step')
        plt.ylabel(f'{target_col} Value')
        plt.title(f'Sample Prediction - {pred_len}h')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Scatter plot
        plt.subplot(1, 3, 3)
        plt.scatter(results['actuals'].flatten(), results['predictions'].flatten(), alpha=0.5)
        plt.plot([results['actuals'].min(), results['actuals'].max()],
                [results['actuals'].min(), results['actuals'].max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Prediction vs Actual - {pred_len}h')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results_{pred_len}h.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: results_{pred_len}h.png")
    
    # Save all results
    with open('forecasting_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n{'='*60}")
    print("All results saved to 'forecasting_results.json'")
    print(f"{'='*60}")
    
    return all_results

# ============================================
# 7. USAGE EXAMPLE
# ============================================
if __name__ == "__main__":
    # Sử dụng với ETTh1
    data_path = r"D:\Transformer\ETTh1.csv"
    
    # Chạy pipeline
    results = run_forecasting_pipeline(
        data_path=data_path,
        pred_lengths=[96, 192, 336, 720],
        target_col='OT',
        epochs=50,
        batch_size=32
    )
    
    # In tổng kết
    print("\n" + "="*60)
    print("SUMMARY OF ALL PREDICTIONS")
    print("="*60)
    for pred_len, metrics in results.items():
        print(f"\n{pred_len}:")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")