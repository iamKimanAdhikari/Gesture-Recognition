from _1combine import Combine
from _setup_logging import SetupLogs
import pandas as pd
from scipy.signal import butter, filtfilt
import pickle
import numpy as np

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
    
    def butterworth_low_pass_filter(self, data, cutoff, fs, order):
        nyquist = 0.5 * fs
        norm_cutoff = cutoff / nyquist
        b, a = butter(order, norm_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data)
    
    def impute_missing_values(self, df):
        """Impute missing values using column median strategy"""
        self.logger.info("Imputing missing values using median strategy")
        for col in df.columns:
            if df[col].isna().any():
                null_count = df[col].isna().sum()
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                self.logger.info(f"Imputed {null_count} missing values in '{col}' with median: {median_val:.4f}")
        return df

    def clean_data(self, cutoff=10, fs=100, order=5):
        self.logger.info("Cleaning combined dataset")
        try:
            # Create a copy of combined data
            self.cleaned_df = self.combined_df.copy()
            
            # 1. Impute missing values
            self.cleaned_df = self.impute_missing_values(self.cleaned_df)
            
            # 2. Apply low-pass filter to finger columns
            filter_columns = ["Index", "Middle", "Ring"]
            for col in filter_columns:
                if col in self.cleaned_df.columns:
                    try:
                        self.cleaned_df[col] = self.butterworth_low_pass_filter(
                            self.cleaned_df[col], cutoff, fs, order
                        )
                        self.logger.info(f"Applied low-pass filter to {col}")
                    except Exception as e:
                        self.logger.error(f"Error filtering {col}: {e}")
            
            # 3. Final null check
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