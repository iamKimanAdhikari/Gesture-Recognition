from ._setup_logging import SetupLogs  
from ._2clean import CleanData        
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

class SplitData(CleanData):
    def __init__(self):
        super().__init__()
        setup = SetupLogs("Split Data")
        self.logger = setup.setup_logging()
        self.cache_path = self.splitted_dir / "_cached_splitted_dfs.pkl"
        
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    segmented_data = pickle.load(f)
                    self.train_df, self.val_df, self.test_df = segmented_data
                self.logger.info("Loaded cached splitted DataFrames from disk.")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        else:
            self.split_data()
    
    def split_data(self):
        self.logger.info("Splitting data into train/val/test sets")
        try:
            train, test_val = train_test_split(
                self.cleaned_df,
                test_size=0.3,
                stratify=self.cleaned_df['Label'],
                random_state=42
            )
            val, test = train_test_split(
                test_val,
                test_size=0.5,
                stratify=test_val['Label'],
                random_state=42
            )
            
            # Save datasets
            train_path = self.splitted_dir / "train.csv"
            val_path = self.splitted_dir / "val.csv"
            test_path = self.splitted_dir / "test.csv"
            
            train.to_csv(train_path, index=False)
            val.to_csv(val_path, index=False)
            test.to_csv(test_path, index=False)
            
            self.train_df = train
            self.val_df = val
            self.test_df = test
            
            # Save cache
            with open(self.cache_path, "wb") as f:
                pickle.dump((train, val, test), f)
                
            self.logger.info(f"Training set size: {len(train)} samples")
            self.logger.info(f"Validation set size: {len(val)} samples")
            self.logger.info(f"Testing set size: {len(test)} samples")
            self.logger.info("Data splitting completed successfully")
            
            
        except Exception as e:
            self.logger.error(f"Error during segmentation: {e}")
            raise