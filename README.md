# 睡眠品質預測模型 (Sleep Quality Prediction Model)

本專案使用 LSTM 神經網路，根據 Fitbit 數據（步數、心率等）預測使用者的睡眠品質。採用 **留一法交叉驗證 (Leave-One-Out Cross-Validation, LOOCV)** 來評估模型的泛化能力。

## 專案結構

- `train.py`: 主要訓練腳本，包含資料處理、模型訓練與評估流程。
- `model.py`: 定義 `PerSQ` 模型架構 (基於 LSTM)。
- `data_loader.py`: 負責讀取與初步處理 CSV 資料。
- `preprocessing.py`: (如果有的話) 用於將原始 PMData 資料集轉換為 `processed_data.csv`。

## 環境設定

請確保已安裝以下 Python 套件：

```bash
pip install numpy pandas torch scikit-learn mlxtend
```

## 訓練流程詳解 (Step-by-Step)

訓練過程主要由 `train.py` 控制，詳細步驟如下：

### 1. 資料載入 (Data Loading)
- 程式首先透過 `DataLoader` 讀取 `processed_data.csv`。
- 根據 `participant_id` 將資料分組，並過濾掉資料量不足（少於視窗大小）的使用者。

### 2. 留一法交叉驗證 (LOOCV)
- 為了模擬真實情境（預測新使用者的睡眠品質），我們使用 LOOCV。
- 假設有 N 位使用者，我們會進行 N 次訓練：
    - 每次選定 **1 位** 使用者作為 **測試集 (Test Set)**。
    - 其餘 **N-1 位** 使用者作為 **訓練集 (Training Set)**。

### 3. 資料前處理與正規化 (Preprocessing & Scaling)
在每一輪 LOOCV 中：
1.  **合併訓練資料**：將 N-1 位訓練使用者的資料合併。
2.  **特徵正規化 (Scaling)**：
    - 使用 `MinMaxScaler` 計算訓練資料的最小值與最大值。
    - 將訓練資料與測試資料的所有特徵縮放到 [0, 1] 之間。
    - **注意**：Scaler 僅由訓練資料擬合 (fit)，以避免資料洩漏 (Data Leakage)。

### 4. 滑動視窗 (Sliding Window)
- 為了讓 LSTM 學習時間序列特徵，我們將資料轉換為視窗序列。
- 預設視窗大小 (`WINDOW_SIZE`) 為 3 天。即利用前 3 天的數據來預測第 3 天（當天）的睡眠品質。
- **重要**：視窗化是針對每位使用者 **獨立進行** 的，不會跨越使用者邊界。

### 5. 模型架構 (Model Architecture)
- 使用定義在 `model.py` 中的 `PerSQ` 模型。
- **架構**：
    - 3 層 LSTM 層 (Hidden Units: 50, 30, 20)
    - Dropout 層 (防止過擬合)
    - 全連接層 (Linear Layer) 輸出最終預測值

### 6. 模型訓練 (Training)
- 使用 Adam 優化器 (Learning Rate = 0.001) 和 MSE Loss (均方誤差) 進行訓練。
- 訓練 50 個 Epochs。

### 7. 評估與還原 (Evaluation & Inverse Transform)
- 模型對測試使用者進行預測。
- 由於預測值是經過正規化的 (0~1)，我們使用之前的 Scaler 將預測值與真實值 **還原 (Inverse Transform)** 回原始尺度 (例如 0-100 分)。
- 計算評估指標：
    - **MAE (平均絕對誤差)**
    - **RMSE (均方根誤差)**
    - **R2 Score (決定係數)**

### 8. 生活型態回饋系統 (Lifestyle Feedback System)
- 系統會利用 **Apriori 演算法** 從「高睡眠品質」的數據中挖掘頻繁模式 (Frequent Patterns)。
- 當模型預測某位使用者的睡眠品質較差時，系統會比對該使用者的特徵與「高睡眠品質模式」。
- 根據差異產生個人化建議 (例如：「你今天的步數偏低，建議多走走」)，並記錄在輸出檔案中。

## 輸出檔案 (Outputs)

訓練完成後，會產生以下檔案：

1.  **`results.csv`**: 包含每位測試使用者的評估指標 (MAE, MSE, RMSE, R2)。
2.  **`detailed_predictions.csv`**: 包含詳細的每日預測結果，欄位如下：
    - `participant_id`: 使用者 ID
    - `date`: 日期
    - `actual_sleep_quality`: 真實睡眠品質
    - `predicted_sleep_quality`: 模型預測的睡眠品質
    - `feedback`: 針對該日期的生活型態改善建議

## 如何執行

在專案根目錄下執行：

```bash
python train.py
```

若使用虛擬環境，請先啟動：

```bash
source .venv/bin/activate
python train.py
```
