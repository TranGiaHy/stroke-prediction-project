import pandas as pd
import numpy as np
import os
def get_raw_data():
    filepath = 'data/raw/healthcare-dataset-stroke-data.csv'
    if not os.path.exists(filepath):
        filepath = '../' + filepath
    df = pd.read_csv(filepath)
    return df