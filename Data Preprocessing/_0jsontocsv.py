import pandas as pd
import json
import numpy as np
import pickle
from _setup_logging import SetupLogs
from _setup_directories import SetupDirectories

class JsontoCsv(SetupDirectories):
    def __init__(self):
        super().__init__()
        setup = SetupLogs("JSON to CSV")
        self.logger = setup.setup_logging()
        # Prepare cache
        self.cache_path = self.raw_csv_dir / "_cached_csv_dfs.pkl"
        self.csv_dfs = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    self.csv_dfs = pickle.load(f)
                self.logger.info("Loaded cached CSV DataFrames from disk.")
            except Exception as e:
                self.logger.warning(f"Failed to load CSV cache: {e}")

    def convert(self):
        try:
            self.json_files = list(self.json_dir.glob("*.json"))
            self.logger.info(f"{len(self.json_files)} .json files found in {self.json_dir}")

            for j in self.json_files:
                stem = j.stem
                if stem in self.csv_dfs:
                    self.logger.info(f"Skipping {stem}, already converted and cached.")
                    continue

                try:
                    with open(j, "r") as file:
                        data = json.load(file)

                    interval_ms = data["payload"]["interval_ms"]
                    values = data["payload"]["values"]
                    parameters = [sensor["name"] for sensor in data["payload"]["sensors"]]
                    timestamps = [i * interval_ms for i in range(len(values))]

                    df = pd.DataFrame(values, columns=parameters)
                    df.insert(0, "timestamps", timestamps)

                    if stem in ["LeftClick", "RightClick"]:
                        mapping = {"Roll": "AccX", "Pitch": "AccY", "Yaw": "AccZ"}
                        for new_col in mapping.values():
                            df[new_col] = 0
                        for orig in mapping.keys():
                            if orig in df.columns:
                                df.drop(columns=[orig], inplace=True)
                        base_name = stem
                    else:
                        parts = stem.split('.')
                        base_name = f"{parts[0]}_{parts[1]}_{parts[3]}"
                    
                    csv_filename = f"{base_name}_raw.csv"
                    output_path = self.raw_csv_dir / csv_filename
                    df.to_csv(output_path, index=False)

                    # Cache the DataFrame
                    self.csv_dfs[stem] = df

                except (KeyError, ValueError, TypeError, IndexError, IOError) as e:
                    self.logger.error(f"Error processing {j}: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error with {j}: {e}")

            # Save cache
            try:
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(self.csv_dfs, f)
                self.logger.info("Cached CSV DataFrames to disk.")
            except Exception as e:
                self.logger.warning(f"Failed to save CSV cache: {e}")

            self.logger.info(f"CSV files saved to {self.raw_csv_dir}")
            total = len(self.csv_dfs)
            self.logger.info(f"{total} files successfully converted and cached.")

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