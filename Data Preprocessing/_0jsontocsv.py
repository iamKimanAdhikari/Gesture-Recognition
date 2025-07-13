import pandas as pd
import re
import json
import logging as log
from pathlib import Path
from setup_logging import SetupLogs

class JsontoCsv:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent

    def setup_directories(self):
        try:
            self.raw_path = self.base_path/"Data"/"Raw_Data"
            self.json_dir = self.raw_path/"jsonRaw"
            self.csv_dir = self.raw_path/"csvRaw"
            self.json_dir.resolve()
            self.csv_dir.resolve()
            self.csv_dir.mkdir(parents = True, exist_ok = True)
            log.info(f"JSON directory: {self.json_dir}")
            log.info(f"CSV directory: {self.csv_dir}")
        except PermissionError as e:
            log.error(f"Insufficient Permission to access the file as {e}")
        except FileNotFoundError as e:
            log.error(f"{self.json_dir} not found as {e}.")
        except Exception as e:
            log.error(f"Unexpected Error occured as {e}")
    #Hello
    def convert(self):
        try:
            self.json_files = list(self.json_dir.glob("*.json"))
            count = len(self.json_files)
            log.info(f"{count} .json files found in {self.json_dir}")
            gesture, code, _, segment, file_type = str(self.json_files[0].name).split(".")
            print(f"{gesture}, {code}, {segment}, {file_type}.")                                  
            for j in self.json_files:
                ...
        except FileNotFoundError as e:
            log.error(f"{self.json_dir} not found.")
        except PermissionError as e:
            log.error(f"Insufficient Permission to access the file as {e}")


# class Combine(JsontoCsv):
#     def __init__(self):
#         ... 

def main():
    setup = SetupLogs("JSON to CSV")
    setup.setup_logging()
    j2c = JsontoCsv()
    j2c.setup_directories()
    j2c.convert()

if __name__ == "__main__":
    main()