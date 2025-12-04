import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_loader import DataLoader
from model import PerSQ
from feedback import FeedbackSystem

# Configuration
DATA_DIR = os.path.join(os.getcwd(), 'pmdata')
PARTICIPANTS = [f'p{i:02d}' for i in range(1, 17)] # p01 to p16
WINDOW_SIZE = 3
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_UNITS = [50, 30, 20]
DROPOUT_RATE = 0.2

def train_model(train_loader, input_size):
    model = PerSQ(input_size, HIDDEN_UNITS, DROPOUT_RATE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
    return model

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions.extend(outputs.view(-1).tolist())
            actuals.extend(y_batch.tolist())
            
    return np.array(predictions), np.array(actuals)

def main():
    print("Initializing Data Loader...")
    dl = DataLoader(DATA_DIR, WINDOW_SIZE)
    
    # Load all data first to handle scaling properly?
    # In LOOCV, we should strictly fit scaler on training set.
    # But for simplicity and speed in this demo, we might load all, 
    # but let's try to do it right: Load all raw dfs, then split.
    
    print("Loading participant data...")
    participant_data = {}
    for p in PARTICIPANTS:
        df = dl.load_participant_data(p)
        if df is not None and len(df) > WINDOW_SIZE:
            participant_data[p] = df
        else:
            print(f"Skipping {p} (insufficient data)")
            
    valid_participants = list(participant_data.keys())
    results = []
    all_predictions = []
    
    # Initialize and Fit Feedback System
    print("Initializing Feedback System...")
    # We need a dataframe with all data to fit thresholds and patterns
    all_data_list = [participant_data[p] for p in valid_participants]
    if all_data_list:
        full_df = pd.concat(all_data_list, ignore_index=True)
        # Identify feature columns (excluding non-features)
        # dl.feature_columns might be empty if we haven't run create_dataset or get_windows
        # Let's manually define or extract
        exclude_cols = ['date', 'sleep_quality', 'participant_id', 'pmsys_sleep_quality']
        feature_cols = [c for c in full_df.columns if c not in exclude_cols]
        
        feedback_sys = FeedbackSystem()
        feedback_sys.fit(full_df, feature_cols, target_col='sleep_quality')
    else:
        print("No data available for Feedback System.")
        feedback_sys = None
    
    print(f"Starting LOOCV on {len(valid_participants)} participants...")
    
    for test_p in valid_participants:
        print(f"Testing on {test_p}...")
        
        # 1. Prepare Train/Test Data
        train_dfs = [participant_data[p] for p in valid_participants if p != test_p]
        test_df = participant_data[test_p]
        
        if not train_dfs:
            continue
            
        train_concat = pd.concat(train_dfs, ignore_index=True)
        
        # 2. Scale Data (Fit on Train, Transform Train & Test)
        # We need to scale features. Target (SQ) is usually 0-100, maybe scale it too?
        # "Output layer maps... to target space v'. ... inverse transformation"
        # Usually better to scale target to 0-1 or similar for NN.
        
        feature_cols = dl.feature_columns
        # If feature_cols is empty, we need to populate it. 
        # It gets populated in create_dataset, but we are using load_participant_data directly.
        # Let's manually get cols from first df.
        if not feature_cols:
             feature_cols = [c for c in test_df.columns if c not in ['date', 'sleep_quality', 'participant_id']]
             feature_cols = ['sleep_quality'] + feature_cols # Include autoregressive SQ
             dl.feature_columns = feature_cols

        # Extract values
        X_train_raw = train_concat[feature_cols].values
        # Target is included in features for autoregression, but we also need it as label.
        # But wait, get_windows extracts X and y.
        # We need to scale the columns in the dataframe or array.
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        
        # We need to apply this scaling to the dataframes before windowing?
        # Or window first then scale? Windowing is just reshaping. Scaling is element-wise.
        # Easier to scale the big matrix, then reshape.
        
        # Reconstruct scaled train df (a bit messy but safe)
        train_concat_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
        train_concat_scaled['sleep_quality_target'] = train_concat['sleep_quality'].values # Keep original target for y?
        # Actually y should also be scaled usually for MSE loss, then inverse transformed.
        # Since 'sleep_quality' is in feature_cols, it is scaled.
        # We can use the scaled 'sleep_quality' as target y.
        
        # Test scaling
        X_test_raw = test_df[feature_cols].values
        X_test_scaled = scaler.transform(X_test_raw)
        test_df_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
        
        # 3. Create Windows
        # We need to handle the boundaries between participants in training set?
        # Yes, we shouldn't window across participants.
        # So we need to window each train participant separately and concat.
        # But we already concatenated them. That's bad for windowing.
        # Let's window each train participant separately.
        
        X_train_all, y_train_all = [], []
        
        # We need to inverse transform the scaler to apply it to individual dfs?
        # No, just use the scaler we fit on the big concat.
        
        # Iterate original train dfs
        for p in valid_participants:
            if p == test_p:
                continue
            p_df = participant_data[p]
            p_vals = p_df[feature_cols].values
            p_vals_scaled = scaler.transform(p_vals)
            
            # Windowing
            # We need a helper that takes array
            # dl.get_windows takes df. Let's make a local helper or modify dl.
            # Let's just do it here.
            
            vals = p_vals_scaled
            # Target is the scaled sleep quality (index 0 if sleep_quality is first)
            target_idx = feature_cols.index('sleep_quality')
            
            X_p, y_p = [], []
            for i in range(WINDOW_SIZE, len(vals)):
                X_p.append(vals[i-WINDOW_SIZE:i])
                y_p.append(vals[i, target_idx])
            
            if X_p:
                X_train_all.append(np.array(X_p))
                y_train_all.append(np.array(y_p))
                
        X_train = np.concatenate(X_train_all)
        y_train = np.concatenate(y_train_all)
        
        # Test Windows
        vals_test = X_test_scaled
        target_idx = feature_cols.index('sleep_quality')
        X_test_list, y_test_list = [], []
        for i in range(WINDOW_SIZE, len(vals_test)):
            X_test_list.append(vals_test[i-WINDOW_SIZE:i])
            y_test_list.append(vals_test[i, target_idx])
            
        if not X_test_list:
            print(f"Skipping {test_p} (insufficient data after windowing)")
            continue
            
        X_test = np.array(X_test_list)
        y_test = np.array(y_test_list)
        
        # 4. Convert to Tensors
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = TorchDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        test_loader = TorchDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 5. Train
        input_size = X_train.shape[2]
        model = train_model(train_loader, input_size)
        
        # 6. Evaluate
        preds_scaled, actuals_scaled = evaluate_model(model, test_loader)
        
        # 7. Inverse Transform
        # We need to inverse transform the target.
        # The scaler was fit on [SQ, Steps, ...].
        # We only have SQ predictions.
        # We need to construct a dummy array to inverse transform, or just manually unscale if we know min/max.
        # MinMaxScaler: X_std = (X - X.min) / (X.max - X.min)
        # X_scaled = X_std * (max - min) + min
        # So X = X_scaled * (X.max - X.min) + X.min
        
        sq_min = scaler.data_min_[target_idx]
        sq_max = scaler.data_max_[target_idx]
        
        preds = preds_scaled * (sq_max - sq_min) + sq_min
        actuals = actuals_scaled * (sq_max - sq_min) + sq_min
        
        # Metrics
        mae = mean_absolute_error(actuals, preds)
        mse = mean_squared_error(actuals, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals, preds)
        
        print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        results.append({
            'participant': test_p,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        })

        # Collect detailed predictions
        # test_df has 'date' column. The predictions correspond to indices [WINDOW_SIZE:]
        if 'date' in test_df.columns:
            dates = test_df['date'].values[WINDOW_SIZE:]
        else:
            # Fallback if date is not available or is index
            dates = test_df.index[WINDOW_SIZE:]
            
        # Ensure lengths match
        min_len = min(len(dates), len(preds), len(actuals))
        
        for i in range(min_len):
            pred_sq = preds[i]
            actual_sq = actuals[i]
            
            # Get the row data corresponding to this prediction
            # The prediction is for date[i]. The input features were from the window ending at date[i].
            # But for feedback, we usually look at the *current* day's behavior that led to this sleep?
            # Or the previous day? "Carry-over effect... adding SQ value in previous day".
            # The paper says "optimize life parameters... to improve individual SQ".
            # Usually this means "What should I do TODAY to sleep well TONIGHT?".
            # Our model predicts SQ for day T based on T-3, T-2, T-1.
            # So if we predict poor sleep for T, it's based on past behavior.
            # Feedback should probably be based on the most recent data (T-1) to suggest changes for T?
            # Or just analyze the pattern of T-1 that led to poor sleep at T.
            
            # Let's assume we want to analyze the features of the last day in the window (T-1)
            # which contributed to the prediction for T.
            # X_test[i] is the window. Shape (WINDOW_SIZE, input_size).
            # The last step in window is X_test[i][-1].
            # But X_test is scaled. We need original values for feedback?
            # Or we can just use the scaled values if we pass scaled df to fit?
            # Wait, we passed `full_df` to `fit`, which is raw (unscaled by train.py, but scaled by preprocessing.py?)
            # `preprocessing.py` does MinMax scaling!
            # So `full_df` is already scaled 0-1.
            # `X_test` is also scaled (re-scaled in train.py? No, we used `scaler` on top of it?
            # train.py: "X_train_scaled = scaler.fit_transform(X_train_raw)"
            # So `train.py` applies a SECOND scaling.
            # `feedback_sys.fit` used `full_df` (First scaling only).
            # So we should use data from `test_df` (First scaling only) to generate feedback.
            
            # Find the row in test_df corresponding to this date
            # date is dates[i].
            current_date = dates[i]
            # We want the data for this date? No, the prediction is FOR this date.
            # The input was previous days.
            # If we want to give feedback "Walk more", it implies changing behavior for the NEXT sleep.
            # But here we are analyzing historical predictions.
            # Let's just analyze the features of the day associated with the prediction.
            # If `test_df` has data for `current_date`, we can use it.
            
            row = test_df[test_df['date'] == current_date].iloc[0] if 'date' in test_df.columns else test_df.iloc[i + WINDOW_SIZE]
            
            feedback_msg = ""
            if feedback_sys:
                feedback_msg = feedback_sys.generate_feedback(row, pred_sq)
            
            all_predictions.append({
                'participant_id': test_p,
                'date': dates[i],
                'actual_sleep_quality': actuals[i],
                'predicted_sleep_quality': pred_sq,
                'feedback': feedback_msg
            })
        
    # Summary
    if results:
        df_results = pd.DataFrame(results)
        print("\nOverall Results:")
        print(df_results.mean(numeric_only=True))
        df_results.to_csv('results.csv', index=False)
        print("Results saved to results.csv")
        
    if all_predictions:
        df_preds = pd.DataFrame(all_predictions)
        df_preds.to_csv('detailed_predictions.csv', index=False)
        print("Detailed predictions saved to detailed_predictions.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
