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
        ...

def main():
    CleanData()

if __name__ == "__main__":
    main()