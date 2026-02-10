import pandas as pd

def load_student_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print("❌ CSV file not found ❌")
        return None
