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
            self.logger.info(f"{len(self.json_files)} .json files found in {self.json_dir}")

            for j in self.json_files:
                try:
                    with open(j, "r") as file:
                        data = json.load(file)

                    interval_ms = data["payload"]["interval_ms"]
                    values = data["payload"]["values"]
                    parameters = [sensor["name"] for sensor in data["payload"]["sensors"]]
                    timestamps = [i * interval_ms for i in range(len(values))]

                    # Build dataframe
                    df = pd.DataFrame(values, columns=parameters)
                    df.insert(0, "timestamps", timestamps)

                    # Only for LeftClick and RightClick, replace Roll, Pitch, Yaw with null AccX, AccY, AccZ
                    if j.stem in ["LeftClick", "RightClick"]:
                        mapping = {"Roll": "AccX", "Pitch": "AccY", "Yaw": "AccZ"}
                        for orig, new_col in mapping.items():
                            if orig in df.columns:
                                df.drop(columns=[orig], inplace=True)
                            df[new_col] = pd.NA

                    # Determine CSV filename
                    if j.stem in ["LeftClick", "RightClick"]:
                        base_name = j.stem
                    else:
                        parts = j.stem.split('.')
                        base_name = f"{parts[0]}{parts[1]}_{parts[3]}"
                    csv_filename = f"{base_name}_raw.csv"
                    output_path = self.raw_csv_dir / csv_filename

                    df.to_csv(output_path, index=False)

                except (KeyError, ValueError, TypeError, IndexError, IOError) as e:
                    self.logger.error(f"Error processing {j}: {e}")
                    raise
                except Exception as e:
                    self.logger.error(f"Unexpected error with {j}: {e}")
                    raise

            self.logger.info(f"CSV files saved to {self.raw_csv_dir}")
            total = len(list(self.raw_csv_dir.glob("*.csv")))
            self.logger.info(f"{total} files successfully converted.")

        except FileNotFoundError:
            self.logger.error("Directory not found.")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("Empty file encountered.")
            raise
        except json.JSONDecodeError:
            self.logger.error("Malformed JSON file.")
            raise
        except Exception as e:
            self.logger.error(str(e))
            raise


def main():
    j2c = JsontoCsv()
    j2c.convert()

if __name__ == "__main__":
    main()
