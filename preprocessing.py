import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def clean_dataset(df):
    df = df.fillna(df.mean())
    scaler = MinMaxScaler()
    numeric_cols = ['HR', 'HRV', 'Sleep_hours', 'Steps', 'Screen_time', 'Typing_speed', 'Attendance']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    df['Stress_level'] = df['Stress_level'].map({'Low': 0, 'Medium': 1, 'High': 2}).astype(int)
    return df

def save_cleaned(df, path="cleaned_dataset.csv"):
    df.to_csv(path, index=False)
    print("Cleaned dataset saved as:", path)
