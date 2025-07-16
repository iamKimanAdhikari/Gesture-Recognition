from _1combine import Combine
from _setup_logging import SetupLogs
from _setup_directories import SetupDirectories
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.signal import butter, filtfilt

class CleanData(Combine):
    def __init__(self):
        super().__init__()
        setup = SetupLogs("Clean Data")
        self.logger = setup.setup_logging()