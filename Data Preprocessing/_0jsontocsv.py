import pandas as pd
import json
import logging as log
from pathlib import Path
from _setup_logging import SetupLogs
from _setup_directories import SetupDirectories

class JsontoCsv(SetupDirectories):
    def __init__(self):
        super().__init__()
        setup = SetupLogs("JSON to CSV")
        setup.setup_logging()

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

                    if j.stem in ["LeftClick", "RightClick"]:
                        base_name = f"{j.stem}"
                    else:
                        gesture = j.stem.split(".")
                        base_name = f"{gesture[0]}{gesture[1]}_{gesture[3]}"
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
    j2c = JsontoCsv()
    j2c.convert()

if __name__ == "__main__":
    main()