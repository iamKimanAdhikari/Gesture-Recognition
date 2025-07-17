import pandas as pd
import json
from _setup_logging import SetupLogs
from _setup_directories import SetupDirectories

class JsontoCsv(SetupDirectories):
    def __init__(self):
        super().__init__()
        setup = SetupLogs("JSON to CSV")
        self.logger = setup.setup_logging()

    def convert(self):
        try:
            self.json_files = list(self.json_dir.glob("*.json"))
            count = len(self.json_files)
            self.logger.info(f"{count} .json files found in {self.json_dir}")

            for j in self.json_files:
                try:
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

                    output_path = self.raw_csv_dir / csv_filename

                    df.to_csv(output_path, index=False)
                    
                except KeyError as e:
                    self.logger.error(f"Invalid key in file {j}: {e}")
                    raise KeyError(f"Invalid key in file {j}: {e}")
                except ValueError as e:
                    self.logger.error(f"Invalid value in the file {j}: {e}")
                    raise ValueError(f"Invalid value in the file {j}: {e}")
                except TypeError as e:
                    self.logger.error(f"Type error in file {j}, check data structures: {e}")
                    raise TypeError(f"Type error in file {j}, check data structures: {e}")
                except IndexError as e:
                    self.logger.error(f"Index error in file {j}, possibly empty lists: {e}")
                    raise IndexError(f"Index error in file {j}, possibly empty lists: {e}")
                except IOError as e:
                    self.logger.error(f"IO error while processing file {j}: {e}")
                    raise IOError(f"IO error while processing file {j}: {e}")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred with file {j}: {e}")
                    raise Exception(f"An unexpected error occurred with file {j}: {e}")
            
            self.logger.info(f"Resulting csv files successfully saved to {self.raw_csv_dir}")
            self.logger.info(f"{len(list(self.raw_csv_dir.glob("*.csv")))} successfully converted.")
        except FileNotFoundError:
            self.logger.error("The directory doesn't exist.")
            raise FileNotFoundError("The directory doesn't exist.")
        except pd.errors.EmptyDataError:
            self.logger.error("The file is empty.")
            raise pd.errors.EmptyDataError("The file is empty.")
        except json.JSONDecodeError:
            self.logger.error("The json file is not in the correct format.")
            raise json.JSONDecodeError("The json file is not in the correct format.")
        except Exception as e:
            self.logger.error(f"{e}")
            raise Exception (f"{e}")


def main():
    j2c = JsontoCsv()
    j2c.convert()

if __name__ == "__main__":
    main()