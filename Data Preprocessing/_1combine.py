import pandas as pd
from _setup_logging import SetupLogs
from _0jsontocsv import JsontoCsv
import pickle

class Combine(JsontoCsv):
    def __init__(self, force_recombine: bool = False):
        super().__init__()
        setup = SetupLogs("Combine")
        self.logger = setup.setup_logging()
        self.force_recombine = force_recombine
        self.cache_path = self.processed_dir / "_cached_combined_df.pkl"
        
        if not self.force_recombine and self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self.combined_df = pickle.load(f)
                self.logger.info("Loaded cached combined DataFrame from disk.")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
                self.combine()
        else:
            self.combine()

    def combine(self):
        self.logger.info(f"Combining relevant files from {self.raw_csv_dir}")
        try:
            master_list = []
            for gesture in self.sorted_gestures:
                last_ts = 0
                segments = []
                for path in self.gestures_dict[gesture]:
                    df = pd.read_csv(path)
                    df["timestamps"] = df["timestamps"] + last_ts
                    last_ts = df["timestamps"].iloc[-1] + df["timestamps"].diff().mean()
                    segments.append(df)
                
                full_df = pd.concat(segments, ignore_index=True)
                master_list.append(full_df)
            
            self.combined_df = pd.concat(master_list, ignore_index=True)
            if "timestamps" in self.combined_df.columns:
                self.combined_df = self.combined_df.drop(columns=["timestamps"])
            
            all_path = self.processed_dir / "allgestures_processed.csv"
            self.combined_df.to_csv(all_path, index=False)
            self.logger.info(f"Saved combined gestures into {all_path}")
            
            # Save cache
            try:
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.combined_df, f)
                self.logger.info("Cached combined DataFrame to disk.")
            except Exception as e:
                self.logger.warning(f"Failed to save cache: {e}")
                
            self.logger.info(f"Successfully combined {len(self.raw_csv_files)} CSVs")
            return self.combined_df
            
        except Exception as e:
            self.logger.error(f"Error during combine: {e}")
            raise