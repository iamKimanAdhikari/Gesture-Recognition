import pandas as pd
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
            raise
        except NotADirectoryError as e:
            log.error(f"A component of the path is a file, but a directory was expected: {e}")
            raise
        except IsADirectoryError as e:
            log.error(f"The path points to a directory, but a file operation was attempted: {e}")
            raise
        except FileNotFoundError as e:
            log.error(f"{self.json_dir} not found as {e}.")
            raise
        except Exception as e:
            log.error(f"Unexpected Error occured as {e}")
            raise

    def convert(self):
        try:
            self.json_files = list(self.json_dir.glob("*.json"))
            count = len(self.json_files)
            log.info(f"{count} .json files found in {self.json_dir}")

            for j in self.json_files:
                try:
                    log.info(f"Processing {j.name} file currently.")
                    with open(j, "r") as file:
                        data = json.load(file)

                    interval_ms = data["payload"]["interval_ms"]
                    values = data["payload"]["values"]
                    parameters = [sensor["name"] for sensor in data["payload"]["sensors"]]
                    timestamps = [i*interval_ms for i in range(len(values))]

                    df = pd.DataFrame(values, columns=parameters)
                    df.insert(0, "timestamps", timestamps)

                    if j.stem == "LeftClick" or "RightClick":
                        base_name = f"{j.stem}"
                    else:
                        gesture, code, _, segment = j.stem.split(".")
                        base_name = f"{gesture}_{code}_{segment}"
                    csv_filename = f"{base_name}_raw.csv"

                    output_path = self.csv_dir / csv_filename

                    df.to_csv(output_path, index=False) 
                    log.info(f"Resulting csv file {csv_filename} successfully saved to {output_path}")
                except KeyError as e:
                    log.error(f"Invalid key in file {j}: {e}")
                    raise
                except ValueError as e:
                    log.error(f"Invalid value in the file {j}: {e}")
                    raise
                except TypeError as e:
                    log.error(f"Type error in file {j}, check data structures: {e}")
                    raise
                except IndexError as e:
                    log.error(f"Index error in file {j}, possibly empty lists: {e}")
                    raise
                except IOError as e:
                    log.error(f"IO error while processing file {j}: {e}")
                    raise
                except Exception as e:
                    log.error(f"An unexpected error occurred with file {j}: {e}")
                    raise

        except FileNotFoundError:
            log.error("The directory doesn't exist.")
            raise
        except pd.errors.EmptyDataError:
            log.error("The file is empty.")
            raise
        except json.JSONDecodeError:
            log.error("The json file is not in the correct format.")
            raise
        except Exception as e:
            log.error(f"An unexpected error occurred: {e}")
            raise


def main():
    setup = SetupLogs("JSON to CSV")
    setup.setup_logging()
    j2c = JsontoCsv()
    j2c.setup_directories()
    j2c.convert()

if __name__ == "__main__":
    main()