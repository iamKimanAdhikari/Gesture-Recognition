from _1combine import Combine
from _setup_logging import SetupLogs
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.signal import butter, filtfilt
from pathlib import Path
import pickle

class CleanData(Combine):
    def __init__(self):
        super().__init__()
        setup = SetupLogs("Clean Data")
        self.logger = setup.setup_logging()
        self.cleaned_dfs = {}

        # Attempt to load cleaned data from cache
        self.cache_path = self.cleaned_dir / "_cached_cleaned_dfs.pkl"
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self.cleaned_dfs = pickle.load(f)
                self.logger.info("Loaded cached cleaned DataFrames from disk.")
            except Exception as e:
                self.logger.warning(f"Failed to load cached cleaned DataFrames: {e}")

    def butterworth_low_pass_filter(self, data, cutoff, fs, order):
        nyquist_frequency = 0.5 * fs
        normal_cutoff = cutoff / nyquist_frequency
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data)

    def remove_outliers(self, cutoff=10, fs=100, order=5):
        try:
            for gesture, df in self.all_dfs.items():
                if gesture in self.cleaned_dfs:
                    self.logger.info(f"Skipping {gesture}, already cleaned and cached.")
                    continue

                self.logger.info(f"Cleaning gesture: {gesture}")

                imputer = SimpleImputer(missing_values=np.nan, strategy="median")
                data_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

                unfiltered_columns = ["Index", "Middle", "Ring"]
                for column in unfiltered_columns:
                    if column in data_imputed.columns:
                        try:
                            filtered_values = self.butterworth_low_pass_filter(
                                data_imputed[column], cutoff, fs, order
                            )
                            data_imputed[column] = filtered_values
                            self.logger.info(f"Successfully filtered column: {column}")
                        except ValueError as e:
                            self.logger.error(f"ValueError filtering {column}: {e}")
                        except TypeError as e:
                            self.logger.error(f"TypeError filtering {column}: {e}")
                        except Exception as e:
                            self.logger.error(f"Unexpected error filtering {column}: {e}")

                self.cleaned_dfs[gesture] = data_imputed

                cleaned_filename = str(self.cleaned_dir / f"{gesture}_cleaned.csv")
                try:
                    data_imputed.to_csv(cleaned_filename, index=False, lineterminator="\n")
                    self.logger.info(f"Saved cleaned data for {gesture} to {cleaned_filename}")
                except PermissionError as e:
                    self.logger.error(f"Permission denied when saving {cleaned_filename}: {e}")
                except FileNotFoundError as e:
                    self.logger.error(f"File path not found when saving {cleaned_filename}: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error saving file {cleaned_filename}: {e}")

            # Save cache
            try:
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.cleaned_dfs, f)
                self.logger.info("Cached cleaned DataFrames to disk.")
            except Exception as e:
                self.logger.warning(f"Failed to save cleaned DataFrame cache: {e}")

        except KeyError as e:
            self.logger.error(f"KeyError in cleaning data: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"ValueError in cleaning data: {e}")
            raise
        except TypeError as e:
            self.logger.error(f"TypeError in cleaning data: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during cleaning process: {e}")
            raise

def main():
    cleaner = CleanData()
    cleaner.remove_outliers()

if __name__ == "__main__":
    main()