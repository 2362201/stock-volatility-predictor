# Stock Volatility Predictor (LSTM + Attention)

Deep learning project to predict stock movements during **extreme volatility/black-swan events**.

## Features
- Downloads 2y SPY data (`yfinance`)
- 13+ engineered features: RSI, Bollinger Bands, VIX proxy, sentiment, trend
- Detects high-volatility regimes & black-swan events
- LSTM + Attention model with RobustScaler
- 5-day directional forecasts + visualizations

## Quick Start
## Internal Output - 
=== STOCK MOVEMENT PREDICTION DURING EXTREME VOLATILITY (COMPLETE) ===
Fetching SPY data...
Data shape: (497, 10)
Training on 463 sequences...
Epoch 1/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 4s 70ms/step - accuracy: 0.5405 - loss: 0.7708 - val_accuracy: 0.5914 - val_loss: 0.6934 - learning_rate: 0.0010
Epoch 2/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step - accuracy: 0.5568 - loss: 0.7218 - val_accuracy: 0.5699 - val_loss: 0.7027 - learning_rate: 0.0010
Epoch 3/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - accuracy: 0.5649 - loss: 0.7002 - val_accuracy: 0.6022 - val_loss: 0.6842 - learning_rate: 0.0010
Epoch 4/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step - accuracy: 0.5324 - loss: 0.7390 - val_accuracy: 0.6022 - val_loss: 0.6612 - learning_rate: 0.0010
Epoch 5/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step - accuracy: 0.5568 - loss: 0.6869 - val_accuracy: 0.5914 - val_loss: 0.6761 - learning_rate: 0.0010
Epoch 6/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - accuracy: 0.5514 - loss: 0.6921 - val_accuracy: 0.5914 - val_loss: 0.6787 - learning_rate: 0.0010
Epoch 7/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step - accuracy: 0.5297 - loss: 0.7180 - val_accuracy: 0.5914 - val_loss: 0.6717 - learning_rate: 0.0010
Epoch 8/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step - accuracy: 0.5649 - loss: 0.6842 - val_accuracy: 0.6129 - val_loss: 0.6707 - learning_rate: 0.0010
Epoch 9/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - accuracy: 0.6108 - loss: 0.6650 - val_accuracy: 0.5914 - val_loss: 0.6776 - learning_rate: 0.0010
Epoch 10/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - accuracy: 0.5784 - loss: 0.6859 - val_accuracy: 0.5806 - val_loss: 0.6744 - learning_rate: 0.0010
Epoch 11/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step - accuracy: 0.5811 - loss: 0.6857 - val_accuracy: 0.5914 - val_loss: 0.6676 - learning_rate: 0.0010
Epoch 12/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - accuracy: 0.5432 - loss: 0.6944 - val_accuracy: 0.5914 - val_loss: 0.6653 - learning_rate: 5.0000e-04
Epoch 13/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - accuracy: 0.5459 - loss: 0.6798 - val_accuracy: 0.6129 - val_loss: 0.6668 - learning_rate: 5.0000e-04
Epoch 14/50
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - accuracy: 0.5784 - loss: 0.6723 - val_accuracy: 0.5914 - val_loss: 0.6694 - learning_rate: 5.0000e-04

=== 5-DAY FORECAST ===
Day 1: 🟢 UP (conf: 30%)
Day 2: 🟢 UP (conf: 29%)
Day 3: 🟢 UP (conf: 28%)
Day 4: 🟢 UP (conf: 29%)
Day 5: 🟢 UP (conf: 30%)<img width="1087" height="429" alt="Screenshot 2025-12-07 150122" src="https://github.com/user-attachments/assets/bfa23184-d4fe-4687-9069-dd1891f7e0c1" />
<img width="580" height="361" alt="image" src="https://github.com/user-attachments/assets/12112bf2-421c-477d-aaac-d89989c05f7c" />

