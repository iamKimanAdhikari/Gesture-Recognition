import pandas as pd
from _setup_logging import SetupLogs
from _get_files import get_files
from _get_gestures import get_gestures
import logging as log
from pathlib import Path

class Combine():
    def __init__(self):
        setup = SetupLogs("Combine CSV")
        setup.setup_logging()
        self.base_path = Path(__file__).parent.parent
        self.csv_dir = self.base_path / "Data" / "Raw_Data" / "csvRaw"
        self.gestures_list = get_gestures(self.csv_dir)
        self.files_dict = get_files(self.csv_dir)
       
    def combine(self):
        log.info(f"Combining relevant files from {self.csv_dir}")
        count = 0
        try:
            for gesture in self.gestures_list:
                current_gesture = gesture

                for file in self.files_dict[gesture]:
                    count += 1
                    print(file)

            print(count)
        except PermissionError as e:
            log.error(f"Insufficient Permission to access the file as {e}")
            raise
        except NotADirectoryError as e:
            log.error(f"A component of the path is a file, but a directory was expected: {e}")
            raise
        except IsADirectoryError as e:
            log.error(f"The path points to a directory, but a file operation was attempted: {e}")
            raise
        except FileNotFoundError as e:
            log.error(f"{self.csv_dir} not found as {e}.")
            raise
        except Exception as e:
            log.error(f"Unexpected Error occured as {e}")
            raise
        
    
def main():
    obj2 = Combine()
    obj2.combine()

if __name__ == "__main__":
    main()