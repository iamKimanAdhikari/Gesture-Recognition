from sklearn.preprocessing import StandardScaler
import pandas as pd
from ._setup_logging import SetupLogs  
from ._3split import SplitData        
import pickle
import numpy as np

class NormalizeData(SplitData):
    def __init__(self):
        super().__init__()
        setup = SetupLogs("Normalize Data")
        self.logger = setup.setup_logging()
        self.cache_path = self.normalized_dir / "_normalized_dfs.pkl"
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
        self.logger.info("Normalizing data")
        try:
            # Separate features and labels
            feature_columns = self.train_df.columns.drop("Label")
            
            # Preserve labels before transformation
            train_labels = self.train_df["Label"].values
            val_labels = self.val_df["Label"].values
            test_labels = self.test_df["Label"].values
            
            # Fit scaler on training data
            self.scaler.fit(self.train_df[feature_columns])
            
            # Transform datasets
            train_features = self.scaler.transform(self.train_df[feature_columns])
            val_features = self.scaler.transform(self.val_df[feature_columns])
            test_features = self.scaler.transform(self.test_df[feature_columns])
            
            # Reconstruct DataFrames with normalized features and original labels
            self.train_df = pd.DataFrame(
                train_features, 
                columns=feature_columns
            )
            self.train_df["Label"] = train_labels
            
            self.val_df = pd.DataFrame(
                val_features, 
                columns=feature_columns
            )
            self.val_df["Label"] = val_labels
            
            self.test_df = pd.DataFrame(
                test_features, 
                columns=feature_columns
            )
            self.test_df["Label"] = test_labels
            
            # Save datasets
            train_path = self.normalized_dir / "train_normalized.csv"
            val_path = self.normalized_dir / "val_normalized.csv"
            test_path = self.normalized_dir / "test_normalized.csv"
            
            self.train_df.to_csv(train_path, index=False)
            self.val_df.to_csv(val_path, index=False)
            self.test_df.to_csv(test_path, index=False)
            
            # Save cache
            with open(self.cache_path, "wb") as f:
                pickle.dump((self.train_df, self.val_df, self.test_df), f)
                
            self.logger.info(f"Normalized data saved to {self.normalized_dir}")
            
        except Exception as e:
            self.logger.error(f"Error during normalization: {e}")
            raise