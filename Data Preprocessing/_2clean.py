from _setup_directories import SetupDirectories
from _setup_logging import SetupLogs
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import pickle

class CleanData(SetupDirectories):
    def __init__(self, combined_data=None):
        super().__init__()
        setup = SetupLogs("Clean Data")
        self.logger = setup.setup_logging()
        self.cleaned_dfs = {}
        self.combined_data = combined_data
        self.cache_path = self.cleaned_dir / "_cached_cleaned_dfs.pkl"
        
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self.cleaned_dfs = pickle.load(f)
                self.logger.info("Loaded cached cleaned DataFrames from disk.")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")

    def butterworth_low_pass_filter(self, data, cutoff, fs, order):
        nyquist = 0.5 * fs
        norm_cutoff = cutoff / nyquist
        b, a = butter(order, norm_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data)

    def check_nulls(self, df, gesture):
        """Check for null values with special handling for click gestures"""
        # Define columns to skip for click gestures
        skip_columns = ["AccX", "AccY", "AccZ"] if gesture in ["LeftClick", "RightClick"] else []
        
        # Check nulls in all columns except skipped ones
        null_counts = df.drop(columns=skip_columns, errors="ignore").isnull().sum()
        total_nulls = null_counts.sum()
        
        if total_nulls > 0:
            self.logger.warning(f"Found {total_nulls} null values in {gesture}:")
            for col, count in null_counts.items():
                if count > 0:
                    self.logger.warning(f"  - Column '{col}': {count} nulls")
            return True
        return False

    def remove_outliers(self, cutoff=10, fs=100, order=5):
        try:
            if self.combined_data is None:
                self.logger.error("No combined data provided!")
                return

            for gesture, df in self.combined_data.items():
                if gesture in self.cleaned_dfs:
                    self.logger.info(f"Skipping {gesture}, already cleaned.")
                    continue

                self.logger.info(f"Cleaning gesture: {gesture}")
                
                # Check for null values before processing
                has_nulls = self.check_nulls(df, gesture)
                if has_nulls:
                    self.logger.warning(f"Null values detected in {gesture} before cleaning")
                
                # Copy DataFrame to preserve original
                cleaned = df.copy()

                # Apply low-pass filter only to specified columns
                filter_columns = ["Index", "Middle", "Ring"]
                for col in filter_columns:
                    if col in cleaned.columns:
                        try:
                            # Skip if column has nulls
                            if cleaned[col].isnull().any():
                                self.logger.warning(f"Skipping filter for {col} due to null values")
                                continue
                                
                            cleaned[col] = self.butterworth_low_pass_filter(
                                cleaned[col], cutoff, fs, order
                            )
                        except Exception as e:
                            self.logger.error(f"Error filtering {col}: {e}")

                # Check for null values after processing
                has_nulls_after = self.check_nulls(cleaned, gesture)
                if has_nulls_after:
                    self.logger.warning(f"Null values detected in {gesture} after cleaning")

                self.cleaned_dfs[gesture] = cleaned
                out_path = self.cleaned_dir / f"{gesture}_cleaned.csv"
                cleaned.to_csv(out_path, index=False)
                self.logger.info(f"Saved cleaned data for {gesture} to {out_path}")

            # Save cache
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cleaned_dfs, f)
            self.logger.info("Cached cleaned DataFrames.")

        except Exception as e:
            self.logger.error(f"Error in remove_outliers: {e}")
            raise

if __name__ == "__main__":
    from _1combine import Combine
    combiner = Combine()
    cleaner = CleanData(combined_data=combiner.all_dfs)
    cleaner.remove_outliers()