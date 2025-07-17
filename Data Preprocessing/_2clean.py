from _1combine import Combine
from _setup_logging import SetupLogs
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.signal import butter, filtfilt

class CleanData(Combine):
    def __init__(self):
        super().__init__()
        setup = SetupLogs("Clean Data")
        self.logger = setup.setup_logging()

    def remove_outliers(self):
        for gesture in self.sorted_gestures:
            print(gesture)
            df = self.all_dfs[gesture]
            print(df.head())

def main():
    clean = CleanData()
    clean.remove_outliers()

if __name__ == "__main__":
    main()