import pandas as pd
import os

DATA_DIR = 'pmdata'
filepath = os.path.join(DATA_DIR, 'participant-overview.xlsx')

if os.path.exists(filepath):
    df = pd.read_excel(filepath)
    print("Columns:", df.columns.tolist())
    print(df.head())
else:
    print("File not found")
