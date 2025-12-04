import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Configuration
DATA_DIR = 'pmdata'
OUTPUT_FILE = 'processed_data.csv'
EXCLUDED_PARTICIPANTS = ['p12']

def load_demographics(filepath):
    """Loads participant demographics from the overview CSV file."""
    # The file has an empty first row, headers on the second row.
    df = pd.read_csv(filepath, header=1)
    return df

def load_fitbit_json(filepath, value_col='value'):
    """Loads a Fitbit JSON file into a DataFrame with a DatetimeIndex."""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if 'dateTime' in df.columns:
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        # Ensure timezone-naive
        if df['dateTime'].dt.tz is not None:
            df['dateTime'] = df['dateTime'].dt.tz_localize(None)
        df = df.set_index('dateTime')
        # Rename value column to something unique based on filename if needed, 
        # but here we return generic and rename later.
        if 'value' in df.columns:
             df[value_col] = pd.to_numeric(df['value'], errors='coerce')
             return df[[value_col]]
    return pd.DataFrame()

def load_pmsys_csv(filepath):
    """Loads PMSYS wellness CSV data."""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    # Assuming there's a date column. Let's inspect standard PMSYS format.
    # Usually 'date' or 'effective_time_frame'.
    # We will look for a column that parses to date.
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        # Ensure timezone-naive
        if df[date_col].dt.tz is not None:
            df[date_col] = df[date_col].dt.tz_localize(None)
        df = df.set_index(date_col)
        return df
    return pd.DataFrame()

def process_participant(p_id, p_dir):
    """Process data for a single participant."""
    print(f"Processing {p_id}...")
    
    # --- Fitbit Data ---
    fitbit_dir = os.path.join(p_dir, 'fitbit')
    
    # 1. Steps
    steps = load_fitbit_json(os.path.join(fitbit_dir, 'steps.json'), 'steps')
    
    # 2. Calories
    calories = load_fitbit_json(os.path.join(fitbit_dir, 'calories.json'), 'calories')
    
    # 3. Distance
    distance = load_fitbit_json(os.path.join(fitbit_dir, 'distance.json'), 'distance')
    
    # 4. Active Minutes (Sedentary, Lightly, Moderately, Very)
    sedentary = load_fitbit_json(os.path.join(fitbit_dir, 'sedentary_minutes.json'), 'sedentary_minutes')
    lightly = load_fitbit_json(os.path.join(fitbit_dir, 'lightly_active_minutes.json'), 'lightly_active_minutes')
    moderately = load_fitbit_json(os.path.join(fitbit_dir, 'moderately_active_minutes.json'), 'moderately_active_minutes')
    very = load_fitbit_json(os.path.join(fitbit_dir, 'very_active_minutes.json'), 'very_active_minutes')
    
    # 5. Sleep (This is usually more complex, often a summary per day. 
    # We'll assume sleep_score.csv or sleep.json has daily summaries)
    # Checking for sleep_score.csv first as it's often cleaner for scores.
    sleep_score_path = os.path.join(fitbit_dir, 'sleep_score.csv')
    if os.path.exists(sleep_score_path):
        sleep = pd.read_csv(sleep_score_path)
        # Assuming 'timestamp' or similar
        if 'timestamp' in sleep.columns:
            sleep['timestamp'] = pd.to_datetime(sleep['timestamp'])
            if sleep['timestamp'].dt.tz is not None:
                sleep['timestamp'] = sleep['timestamp'].dt.tz_localize(None)
            sleep = sleep.set_index('timestamp')
            # Keep relevant columns like 'overall_score', 'efficiency' if available
            # If columns are not known, we might need to inspect. 
            # For now, let's assume we want 'overall_score' as a proxy for SQ if available, 
            # or we might need to calculate efficiency from sleep.json.
            # Let's try to keep all numeric columns for now.
            sleep = sleep.select_dtypes(include=[np.number])
    else:
        sleep = pd.DataFrame()

    # Merge Fitbit Data (resample to daily sum for activity, mean for others if needed)
    # Most Fitbit JSONs are already daily or minute-level. 
    # Steps/Calories/Distance/ActiveMinutes are usually daily summaries in these JSONs for pmdata?
    # Actually, pmdata often has minute-level data in these JSONs. We need to resample.
    
    daily_dfs = []
    
    for name, df, agg in [
        ('steps', steps, 'sum'),
        ('calories', calories, 'sum'),
        ('distance', distance, 'sum'),
        ('sedentary_minutes', sedentary, 'sum'),
        ('lightly_active_minutes', lightly, 'sum'),
        ('moderately_active_minutes', moderately, 'sum'),
        ('very_active_minutes', very, 'sum')
    ]:
        if not df.empty:
            # Resample to daily
            daily = df.resample('D').agg(agg)
            daily_dfs.append(daily)
            
    if not sleep.empty:
        # Sleep is usually one per night, effectively daily.
        # Ensure it's indexed by date (start of sleep or end of sleep).
        # We'll resample to D just in case.
        daily_sleep = sleep.resample('D').mean() 
        daily_dfs.append(daily_sleep)

    if not daily_dfs:
        return pd.DataFrame()

    # Merge all daily fitbit data
    fitbit_daily = pd.concat(daily_dfs, axis=1)
    
    # --- PMSYS Data ---
    pmsys_dir = os.path.join(p_dir, 'pmsys')
    wellness = load_pmsys_csv(os.path.join(pmsys_dir, 'wellness.csv'))
    
    if not wellness.empty:
        # Resample to daily (though usually it is daily)
        wellness_daily = wellness.resample('D').mean(numeric_only=True)
        
        # Ensure index names match for join
        fitbit_daily.index.name = 'date'
        wellness_daily.index.name = 'date'
        
        print(f"Fitbit Index: {fitbit_daily.index.name}, Type: {type(fitbit_daily.index)}")
        print(f"Wellness Index: {wellness_daily.index.name}, Type: {type(wellness_daily.index)}")
        
        # Merge with Fitbit using pd.merge for robustness
        full_daily = pd.merge(fitbit_daily, wellness_daily, left_index=True, right_index=True, how='outer')
    else:
        full_daily = fitbit_daily
        
    full_daily['participant_id'] = p_id
    return full_daily

def main():
    print("Starting preprocessing...")
    
    # 1. Load Demographics
    demographics_path = os.path.join(DATA_DIR, 'participant-overview.csv')
    if os.path.exists(demographics_path):
        demographics = load_demographics(demographics_path)
        # Select relevant columns: Participant ID, Age, Gender, Height
        # Check column names from inspection: "Participant ID", "Age", "Height", "Gender"
        # Clean column names (strip whitespace)
        demographics.columns = demographics.columns.str.strip()
        
        cols_to_keep = ['Participant ID', 'Age', 'Gender', 'Height']
        # Filter if they exist
        cols_to_keep = [c for c in cols_to_keep if c in demographics.columns]
        demographics = demographics[cols_to_keep]
    else:
        print("Warning: Demographics file not found.")
        demographics = pd.DataFrame()

    all_data = []
    
    # 2. Iterate Participants
    participants = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('p')]
    
    for p in participants:
        if p in EXCLUDED_PARTICIPANTS:
            continue
            
        p_dir = os.path.join(DATA_DIR, p)
        p_data = process_participant(p, p_dir)
        
        if not p_data.empty:
            all_data.append(p_data)
            
    if not all_data:
        print("No data found.")
        return

    # 3. Concatenate all
    final_df = pd.concat(all_data)
    
    # 4. Merge Demographics
    if not demographics.empty:
        # Rename 'Participant ID' to 'participant_id'
        if 'Participant ID' in demographics.columns:
            demographics = demographics.rename(columns={'Participant ID': 'participant_id'})
            
        # Merge
        if 'participant_id' in demographics.columns:
            # Ensure participant_id matches format (p01, etc.) - it does in CSV.
            final_df = pd.merge(final_df, demographics, on='participant_id', how='left')
            
    # 5. Feature Engineering
    # - Handle missing values
    final_df = final_df.fillna(method='ffill').fillna(0) 
    
    # - One-Hot Encoding for categorical columns (e.g. Gender)
    # Identify categorical columns (excluding participant_id, date)
    cat_cols = final_df.select_dtypes(include=['object', 'category']).columns
    cat_cols = [c for c in cat_cols if c not in ['participant_id', 'date', 'timestamp']]
    
    if cat_cols:
        print(f"One-Hot Encoding columns: {cat_cols}")
        final_df = pd.get_dummies(final_df, columns=cat_cols)
    
    # - Normalization
    # Select numeric columns (excluding date, participant_id)
    # We should exclude 'sleep_quality' from scaling? 
    # The requirement says "Min-Max normalization... scale ALL data to [0, 1]".
    # So yes, scale everything.
    
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns
    # Exclude participant_id if it somehow became numeric (unlikely with 'p01')
    
    print(f"Normalizing {len(numeric_cols)} columns...")
    scaler = MinMaxScaler()
    final_df[numeric_cols] = scaler.fit_transform(final_df[numeric_cols])
    
    # 6. Save
    final_df.to_csv(OUTPUT_FILE)
    print(f"Saved processed data to {OUTPUT_FILE}")
    print(f"Shape: {final_df.shape}")
    print(f"Columns: {final_df.columns.tolist()}")

if __name__ == "__main__":
    main()
