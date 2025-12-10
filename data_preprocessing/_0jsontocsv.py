import pandas as pd
import json
import numpy as np
import pickle
from ._setup_logging import SetupLogs  
from ._get_files import GetFiles 

class JsontoCsv(GetFiles):
    def __init__(self):
        super().__init__()
        setup = SetupLogs("JSON to CSV")
        self.logger = setup.setup_logging()
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
                    self.logger.info(f"Skipping {stem}, already converted.")
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

                    # --- LOGIC FOR CONTINUOUS FILES (Left/Right Click) ---
                    if stem in ["LeftClick", "RightClick"]:
                        # 1. Map columns and clean
                        mapping = {"Roll": "AccX", "Pitch": "AccY", "Yaw": "AccZ"}
                        for new_col in mapping.values():
                            df[new_col] = np.nan
                        for orig in mapping.keys():
                            if orig in df.columns:
                                df.drop(columns=[orig], inplace=True)
                        
                        CHUNK_SIZE = 100 
                        
                        total_samples = len(df)
                        num_chunks = int(np.ceil(total_samples / CHUNK_SIZE))
                        
                        self.logger.info(f"Splitting {stem} ({total_samples} rows) into {num_chunks} files.")

                        for i in range(num_chunks):
                            start_idx = i * CHUNK_SIZE
                            end_idx = start_idx + CHUNK_SIZE
                            
                            chunk_df = df.iloc[start_idx:end_idx].copy()
                            
                            # Reset time for the new file
                            if not chunk_df.empty:
                                start_time = chunk_df['timestamps'].iloc[0]
                                chunk_df['timestamps'] = chunk_df['timestamps'] - start_time
                            
                            # Generate Unique ID
                            unique_id = f"{stem}_part{i+1:03d}" 
                            chunk_df["Label"] = stem
                            chunk_df["RecordingID"] = unique_id
                            
                            csv_filename = f"{unique_id}_raw.csv"
                            output_path = self.raw_csv_dir / csv_filename
                            chunk_df.to_csv(output_path, index=False)
                            
                            # Update Cache
                            self.csv_dfs[unique_id] = chunk_df
                            
                        # Mark main stem as processed
                        self.csv_dfs[stem] = "Processed_Split"

                    # --- LOGIC FOR STANDARD FILES ---
                    else:
                        parts = stem.split('.')
                        base_name = f"{parts[0]}_{parts[1]}_{parts[3]}"
                        recordingID = f"{parts[0]}_{parts[1]}"
                        
                        df["Label"] = parts[0]
                        df["RecordingID"] = recordingID
                        
                        csv_filename = f"{base_name}_raw.csv"
                        output_path = self.raw_csv_dir / csv_filename
                        df.to_csv(output_path, index=False)

                        self.csv_dfs[stem] = df

                except Exception as e:
                    self.logger.error(f"Error processing {j}: {e}")

            # Save final cache
            try:
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(self.csv_dfs, f)
            except Exception as e:
                self.logger.warning(f"Failed to save CSV cache: {e}")

            self.logger.info(f"CSV files saved to {self.raw_csv_dir}")

        except Exception as e:
            self.logger.error(str(e))
            raise