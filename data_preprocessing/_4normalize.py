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
        
        # We always run normalization to ensure the scaler is fresh and matches current data
        self.normalize_data()

    def _transform_helper(self, df, scaler, cols):
        """
        Helper method to transform specific columns and rebuild the DataFrame.
        Preserves non-scaled columns (Label, RecordingID, etc.)
        """
        if df is None or df.empty:
            return df
        
        # 1. Separate metadata (columns we don't scale)
        # We use drop to keep everything EXCEPT the feature columns
        metadata = df.drop(columns=cols)
        
        # 2. Transform features
        # This returns a numpy array
        scaled_features = scaler.transform(df[cols])
        
        # 3. Create temporary DF for scaled features
        # We must use the original index to ensure concat works correctly
        scaled_df = pd.DataFrame(
            scaled_features, 
            columns=cols, 
            index=df.index
        )
        
        # 4. Concatenate side-by-side
        result = pd.concat([metadata, scaled_df], axis=1)
        
        # 5. Return with original column order
        return result[df.columns]

    def normalize_data(self):
        self.logger.info("Normalizing data")
        try:
            # Safety Check
            if self.train_df is None:
                raise ValueError("self.train_df is None. Please delete cache in Splitted_Data and rerun.")

            # 1. Identify Feature Columns (Exclude Metadata)
            # We explicitly exclude these columns from scaling
            exclude_cols = ["Label", "RecordingID", "timestamps", "Index"]
            
            # Select only numeric columns that are NOT in the exclude list
            feature_columns = [
                c for c in self.train_df.columns 
                if c not in exclude_cols and pd.api.types.is_numeric_dtype(self.train_df[c])
            ]
            
            if not feature_columns:
                raise ValueError("No feature columns found to normalize!")

            self.logger.info(f"Features being scaled: {feature_columns}")
            
            # 2. Fit Scaler ONLY on Training Data
            self.scaler.fit(self.train_df[feature_columns])
            
            # 3. Transform datasets using the helper
            # This updates the class attributes with the new, normalized DataFrames
            self.train_df = self._transform_helper(self.train_df, self.scaler, feature_columns)
            self.val_df = self._transform_helper(self.val_df, self.scaler, feature_columns)
            self.test_df = self._transform_helper(self.test_df, self.scaler, feature_columns)
            
            # 4. Save CSVs
            # (If self.train_df was None, the code would have crashed earlier, so this is safe now)
            self.train_df.to_csv(self.normalized_dir / "train_normalized.csv", index=False)
            self.val_df.to_csv(self.normalized_dir / "val_normalized.csv", index=False)
            self.test_df.to_csv(self.normalized_dir / "test_normalized.csv", index=False)
            
            # 5. Save scaler (Critical for Inference/ESP32)
            scaler_path = self.normalized_dir / "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            
            # 6. Cache DFs
            with open(self.cache_path, "wb") as f:
                pickle.dump((self.train_df, self.val_df, self.test_df), f)
                
            self.logger.info(f"Normalized data saved to {self.normalized_dir}")
            self.logger.info(f"Scaler saved to {scaler_path}")
            
        except Exception as e:
            self.logger.error(f"Error during normalization: {e}")
            # Raising the error helps you see the stack trace in the terminal
            raise