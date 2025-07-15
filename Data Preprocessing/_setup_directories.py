from pathlib import Path
import logging as log

class SetupDirectories:
    def __init__(self):
        try:
            self.data_dir = Path(__file__).parent.parent/"Data"

            self.raw_dir = self.data_dir / "Raw_Data"
            self.json_dir = self.raw_dir/ "jsonRaw"
            self.csv_dir = self.raw_dir/ "csvRaw"

            self.processed_dir = self.data_dir / "Processed_Data"
            self.cleaned_dir = self.data_dir / "Cleaned_Data"

            self.json_dir.resolve()
            self.csv_dir.resolve()            
            self.processed_dir.resolve()            
            
            for dir in [self.csv_dir,self.processed_dir, self.cleaned_dir]:
                dir.mkdir(parents=True, exist_ok=True)

        except PermissionError as e:
            log.error(f"Insufficient Permission to access the file as {e}")
            raise PermissionError(f"Insufficient Permission to access the file as {e}")
        except FileNotFoundError as e:
            log.error(f"{self.json_dir} not found as {e}.")
            raise FileNotFoundError(f"{self.json_dir} not found as {e}.")
        except Exception as e:
            log.error(f"Unexpected Error occured as {e}")
            raise Exception(f"Unexpected Error occured as {e}")