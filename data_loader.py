import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    def __init__(self, data_dir, window_size=3):
        self.data_dir = data_dir
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        
        # Load processed data
        # Assuming data_dir is the 'pmdata' folder, but processed_data.csv is in 'project' root usually?
        # preprocessing.py saved it to 'processed_data.csv' in current working directory (project root).
        # train.py sets DATA_DIR = os.path.join(os.getcwd(), 'pmdata').
        # So if I pass DATA_DIR to DataLoader, it points to pmdata.
        # But processed_data.csv is likely in the parent of pmdata (project root).
        # Let's check where preprocessing.py saved it.
        # OUTPUT_FILE = 'processed_data.csv'. It ran in project root. So it's ./processed_data.csv.
        # train.py passes DATA_DIR='.../pmdata'.
        # So we should look one level up or just expect it in the same dir as the script?
        # Let's try to find it.
        
        self.project_root = os.path.dirname(data_dir) # Assuming data_dir is .../project/pmdata
        self.processed_file = os.path.join(self.project_root, 'processed_data.csv')
        
        # Fallback: check if it's inside data_dir
        if not os.path.exists(self.processed_file):
            self.processed_file = os.path.join(data_dir, 'processed_data.csv')
            
        # Fallback: check current working directory
        if not os.path.exists(self.processed_file):
            self.processed_file = 'processed_data.csv'

        if os.path.exists(self.processed_file):
            print(f"Loading data from {self.processed_file}...")
            self.full_data = pd.read_csv(self.processed_file)
            if 'date' in self.full_data.columns:
                self.full_data['date'] = pd.to_datetime(self.full_data['date'])
        else:
            self.full_data = pd.DataFrame()
            print(f"Warning: {self.processed_file} not found. Please run preprocessing.py first.")

    def load_participant_data(self, participant_id):
        if self.full_data.empty:
            return None
            
        df = self.full_data[self.full_data['participant_id'] == participant_id].copy()
        
        if df.empty:
            return None
            
        # Rename columns to match train.py expectations
        # processed_data.csv has 'sleep_quality' (PMSYS) and 'overall_score' (Fitbit)
        # We want to use Fitbit 'overall_score' as the target 'sleep_quality'
        
        if 'sleep_quality' in df.columns:
            df = df.rename(columns={'sleep_quality': 'pmsys_sleep_quality'})
            
        if 'overall_score' in df.columns:
            df = df.rename(columns={'overall_score': 'sleep_quality'})
        
        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        return df

    def create_dataset(self, participant_ids):
        all_data = []
        
        for pid in participant_ids:
            df = self.load_participant_data(pid)
            if df is None or len(df) < self.window_size + 1:
                continue
            
            # Define feature columns if not already defined
            if not self.feature_columns:
                # Exclude non-feature columns
                exclude_cols = ['date', 'sleep_quality', 'participant_id']
                self.feature_columns = ['sleep_quality'] + [c for c in df.columns if c not in exclude_cols]
            
            # Note: The data in processed_data.csv is already scaled (MinMax).
            # However, train.py might attempt to scale it again. 
            # Since MinMax(MinMax(x)) = MinMax(x) if range is [0,1], it should be fine.
            
            df['participant_id'] = pid
            all_data.append(df)
            
        if not all_data:
            return pd.DataFrame()
            
        return pd.concat(all_data, ignore_index=True)

    def get_windows(self, df, window_size=3):
        # Assumes df is for a SINGLE participant and sorted by date
        X = []
        y = []
        
        if not self.feature_columns:
             # Fallback if feature_columns not set
             exclude_cols = ['date', 'sleep_quality', 'participant_id']
             self.feature_columns = ['sleep_quality'] + [c for c in df.columns if c not in exclude_cols]

        # Features
        # Ensure all feature columns exist
        valid_features = [c for c in self.feature_columns if c in df.columns]
        features = df[valid_features].values
        target = df['sleep_quality'].values
        
        for i in range(window_size, len(df)):
            # Input: i-window_size to i-1
            # Target: i
            X.append(features[i-window_size:i])
            y.append(target[i])
            
        return np.array(X), np.array(y)
