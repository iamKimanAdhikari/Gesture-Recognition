from ._setup_logging import SetupLogs  
from ._1combine import Combine        
import pickle
import numpy as np
import pandas as pd

class CleanData(Combine):
    def __init__(self):
        super().__init__()
        setup = SetupLogs("Clean Data")
        self.logger = setup.setup_logging()
        self.cache_path = self.cleaned_dir / "_cached_cleaned_df.pkl"
        
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self.cleaned_df = pickle.load(f)
                self.logger.info("Loaded cached cleaned DataFrame from disk.")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        else:
            self.clean_data()
    
    
    # def impute_missing_values(self, df):
    #     self.logger.info("Imputing missing values")
        
    #     numeric_cols = df.select_dtypes(include=[np.number]).columns
        
    #     for col in numeric_cols:
    #         if df[col].isna().any():
    #             null_count = df[col].isna().sum()
    #             median_val = df[col].median()
    #             df[col].fillna(median_val, inplace=True)
    #             self.logger.info(f"Imputed {null_count} missing values in '{col}' with median: {median_val:.4f}")
    #     return df

    def clean_data(self):
        self.logger.info("Cleaning combined dataset")
        try:
            self.cleaned_df = self.combined_df.copy()
            
            # self.cleaned_df = self.impute_missing_values(self.cleaned_df)
            
            null_counts = self.cleaned_df.isnull().sum()
            total_nulls = null_counts.sum()
            if total_nulls > 0:
                self.logger.warning(f"Found {total_nulls} null values after cleaning")
                for col, count in null_counts.items():
                    if count > 0:
                        self.logger.warning(f"  - Column '{col}': {count} nulls")
            else:
                self.logger.info("No null values found after cleaning")
            
            # Save cleaned data
            out_path = self.cleaned_dir / "allgestures_cleaned.csv"
            self.cleaned_df.to_csv(out_path, index=False)
            self.logger.info(f"Saved cleaned data to {out_path}")
            
            # Save cache
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cleaned_df, f)
            self.logger.info("Cached cleaned DataFrame")
            
            return self.cleaned_df
            
        except Exception as e:
            self.logger.error(f"Error in cleaning: {e}")
            raise