from ._setup_logging import SetupLogs  
from ._2clean import CleanData        
import pandas as pd
import pickle
from sklearn.model_selection import GroupShuffleSplit

class SplitData(CleanData):
    def __init__(self):
        super().__init__()
        setup = SetupLogs("Split Data")
        self.logger = setup.setup_logging()
        self.cache_path = self.splitted_dir / "_cached_splitted_dfs.pkl"
        
        self.split_data()
    
    def split_data(self):
        self.logger.info("Splitting data by RecordingID (Time-Series Safe)")
        try:
            df = self.cleaned_df.copy()
            
            # Verify RecordingID exists
            if "RecordingID" not in df.columns:
                self.logger.error("RecordingID column missing! Cannot split safely.")
                raise KeyError("RecordingID column missing")

            # SPLIT 1: Train vs (Test + Val)
            # GroupShuffleSplit ensures all 's1..s15' of a specific recording stay together
            splitter = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
            train_idx, test_val_idx = next(splitter.split(df, groups=df['RecordingID']))
            
            train = df.iloc[train_idx]
            test_val = df.iloc[test_val_idx]

            # SPLIT 2: Val vs Test
            splitter_val = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=42)
            val_idx, test_idx = next(splitter_val.split(test_val, groups=test_val['RecordingID']))
            
            val = test_val.iloc[val_idx]
            test = test_val.iloc[test_idx]

            # Save datasets
            train.to_csv(self.splitted_dir / "train.csv", index=False)
            val.to_csv(self.splitted_dir / "val.csv", index=False)
            test.to_csv(self.splitted_dir / "test.csv", index=False)
            
            self.train_df = train
            self.val_df = val
            self.test_df = test
            
            # Save cache
            with open(self.cache_path, "wb") as f:
                pickle.dump((train, val, test), f)
                
            self.logger.info(f"Training set: {len(train)} rows")
            self.logger.info(f"Validation set: {len(val)} rows")
            self.logger.info(f"Testing set: {len(test)} rows")
            self.logger.info(f"Unique Recordings in Train: {train['RecordingID'].nunique()}")
            self.logger.info("Data splitting completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during splitting: {e}")
            raise