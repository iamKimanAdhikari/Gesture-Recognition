import pandas as pd
from _setup_logging import SetupLogs
from _get_files import get_files
from _get_gestures import get_gestures
from _setup_directories import SetupDirectories

class Combine(SetupDirectories):
    def __init__(self):         
        super().__init__()
        setup = SetupLogs("Combine")
        self.logger = setup.setup_self.logger()
        self.files_dict = get_files()
        self.gestures = get_gestures()
        self.all_dfs = {}
       
    def combine(self):
        self.logger.info(f"Combining relevant files from {self.csv_dir}")
        try:
            for gesture in self.gestures:
                self.logger.info(f"Processing gesture: {gesture}")
                
                # Reset for each gesture
                last_ts = 0
                segments = []
                for i in range(len(self.files_dict[gesture])):
                    df = pd.read_csv(self.files_dict[gesture][i])
                    df["timestamps"] = df["timestamps"] + last_ts
                    last_ts = df["timestamps"].iloc[-1]
                    segments.append(df)

                full_df = pd.concat(segments, ignore_index=True)
                self.all_dfs[gesture] = full_df
                new_name = self.processed_dir/(gesture+"_processed.csv")
                full_df.to_csv(new_name, index = False)
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
    combine.combine()

if __name__ == "__main__":
    main()