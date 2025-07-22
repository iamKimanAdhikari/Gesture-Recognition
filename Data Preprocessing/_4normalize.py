from sklearn.preprocessing import StandardScaler
import pandas as pd
from _3split import SplitData
from _setup_logging import SetupLogs
import pickle

class NormalizeData(SplitData):
    def __init__(self):
        super().__init__()
        setup = SetupLogs("Normalize Data")
        self.logger = setup.setup_logging()
        self.cache_path = self.splitted_dir / "_normalized_dfs.pkl"
        self.scaler = StandardScaler()
        
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    segmented_data = pickle.load(f)
                    self.train_df, self.val_df, self.test_df = segmented_data
                self.logger.info("Loaded cached normalized DataFrames from disk.")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        else:
            self.normalize_data()

    def normalize_data(self):
        ...
        