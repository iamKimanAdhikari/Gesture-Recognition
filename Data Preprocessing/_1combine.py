import pandas as pd
from _setup_logging import SetupLogs
from _get_files import GetFiles
from collections import defaultdict
import pickle

class Combine(GetFiles):
    def __init__(self, force_recombine: bool = False):
        super().__init__()
        setup = SetupLogs("Combine")
        self.logger = setup.setup_logging()
        self.all_dfs = defaultdict(pd.DataFrame)
        self.force_recombine = force_recombine

        # Path to cache
        self.cache_path = self.processed_dir / "_cached_combined_dfs.pkl"
        loaded = False

        # Try loading cache if not forcing recombine
        if self.cache_path.exists() and not self.force_recombine:
            try:
                with open(self.cache_path, "rb") as f:
                    self.all_dfs = pickle.load(f)
                self.logger.info("Loaded cached combined DataFrames from disk.")
                loaded = True
            except Exception as e:
                self.logger.warning(f"Failed to load cached combined DataFrames: {e}")

        # If cache not loaded or forcing recombine, run combine()
        if not loaded:
            self.combine()

    def combine(self):
        # Skip if already combined and not forcing recombine
        if self.all_dfs and not self.force_recombine:
            self.logger.info("all_dfs already populated; skipping combine().")
            return self.all_dfs

        self.logger.info(f"Combining relevant files from {self.raw_csv_dir}")
        try:
            for gesture in self.sorted_gestures:
                last_ts = 0
                segments = []
                for path in self.gestures_dict[gesture]:
                    df = pd.read_csv(path)
                    df["timestamps"] = df["timestamps"] + last_ts
                    last_ts = df["timestamps"].iloc[-1]
                    segments.append(df)

                full_df = pd.concat(segments, ignore_index=True)
                self.all_dfs[gesture] = full_df
                output_path = self.processed_dir / f"{gesture}_processed.csv"
                full_df.to_csv(output_path, index=False)

            # Save cache
            try:
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.all_dfs, f)
                self.logger.info("Cached combined DataFrames to disk.")
            except Exception as e:
                self.logger.warning(f"Failed to save combined DataFrame cache: {e}")

            self.logger.info(
                f"Successfully combined {len(list(self.raw_csv_dir.glob('*.csv')))} CSV files into {len(self.all_dfs)} processed files.")
            return self.all_dfs

        except PermissionError as e:
            self.logger.error(f"Insufficient Permission to access the file: {e}")
            raise
        except NotADirectoryError as e:
            self.logger.error(f"Expected directory but got file: {e}")
            raise
        except IsADirectoryError as e:
            self.logger.error(f"Expected file but got directory: {e}")
            raise
        except FileNotFoundError as e:
            self.logger.error(f"Path not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error occurred: {e}")
            raise


def main():
    # Pass force_recombine=True to bypass cache
    combiner = Combine()

if __name__ == "__main__":
    main()
