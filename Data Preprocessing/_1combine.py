import pandas as pd
from _setup_logging import SetupLogs
from _get_files import GetFiles
from collections import defaultdict
class Combine(GetFiles):
    def __init__(self):         
        super().__init__()
        setup = SetupLogs("Combine")
        self.logger = setup.setup_logging()
        self.all_dfs = defaultdict(pd.DataFrame)

        self.logger.info(f"Combining relevant files from {self.raw_csv_dir}")
        try:
            for gesture in self.sorted_gestures:
                last_ts = 0
                segments = []
                for i in range(len(self.gestures_dict[gesture])):
                    df = pd.read_csv(self.gestures_dict[gesture][i])
                    df["timestamps"] = df["timestamps"] + last_ts
                    last_ts = df["timestamps"].iloc[-1]
                    segments.append(df)

                full_df = pd.concat(segments, ignore_index=True)
                self.all_dfs[f"{gesture}"] = full_df
                new_name = self.processed_dir/(gesture+"_processed.csv")
                full_df.to_csv(new_name, index = False)
            
            self.logger.info(f"Successfully combined {len(list(self.raw_csv_dir.glob("*.csv")))} csv files to {len(list(self.processed_dir.glob("*.csv")))} files.")

        except PermissionError as e:
            self.logger.error(f"Insufficient Permission to access the file as {e}")
            raise PermissionError(f"Insufficient Permission to access the file as {e}")
        except NotADirectoryError as e:
            self.logger.error(f"A component of the path is a file, but a directory was expected: {e}")
            raise NotADirectoryError(f"A component of the path is a file, but a directory was expected: {e}")
        except IsADirectoryError as e:
            self.logger.error(f"The path points to a directory, but a file operation was attempted: {e}")
            raise IsADirectoryError(f"The path points to a directory, but a file operation was attempted: {e}")
        except FileNotFoundError as e:
            self.logger.error(f"{self.csv_dir} not found as {e}.")
            raise FileNotFoundError(f"{self.csv_dir} not found as {e}.")
        except Exception as e:
            self.logger.error(f"Unexpected Error occured as {e}")
            raise Exception(f"Unexpected Error occured as {e}")

def main():
    combine = Combine()
if __name__ == "__main__":
    main()

