import pandas as pd
from _setup_logging import SetupLogs
from _get_files import GetFiles
from collections import defaultdict
import pickle
import os

class Combine(GetFiles):
    def __init__(self, force_recombine: bool = False):
        super().__init__()
        setup = SetupLogs("Combine")
        self.logger = setup.setup_logging()
        self.all_dfs = defaultdict(pd.DataFrame)
        self.force_recombine = force_recombine
        self.cache_path = self.processed_dir / "_cached_combined_dfs.pkl"
        
        if not self.force_recombine and self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self.all_dfs = pickle.load(f)
                self.logger.info("Loaded cached combined DataFrames from disk.")
            except Exception as e:
                self.logger.warning(f"Failed to load cached combined DataFrames: {e}")
                self.combine()
        else:
            self.combine()

    def combine(self):
        self.logger.info(f"Combining relevant files from {self.raw_csv_dir}")
        try:
            for gesture in self.sorted_gestures:
                last_ts = 0
                segments = []
                for path in self.gestures_dict[gesture]:
                    df = pd.read_csv(path)
                    df["timestamps"] = df["timestamps"] + last_ts
                    last_ts = df["timestamps"].iloc[-1] + df["timestamps"].diff().mean()
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
                f"Successfully combined {len(self.raw_csv_files)} CSV files into {len(self.all_dfs)} processed files.")
            return self.all_dfs

        except Exception as e:
            self.logger.error(f"Error during combine: {e}")
            raise


def main():
    combiner = Combine(force_recombine=True)

if __name__ == "__main__":
    main()