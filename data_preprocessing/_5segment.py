from ._setup_logging import SetupLogs
from ._4normalize import NormalizeData
import numpy as np
import pandas as pd
import pickle

class SegmentData(NormalizeData):
    def __init__(self, window_size=100, step_size=50):
        self.window_size = window_size
        self.step_size = step_size

        super().__init__()
        setup = SetupLogs("Segment Data")
        self.logger = setup.setup_logging()

        self.cache_path = self.segmented_dir / "_cached_segmented.pkl"

        # Always regenerate segments – usually safer
        self.segment()

    def _segment_one(self, df, feature_cols):
        segments = []
        labels = []
        recording_ids = []

        # group by RecordingID, preserving time order
        grouped = df.sort_values("timestamps").groupby("RecordingID")

        for rid, group in grouped:
            values = group[feature_cols].to_numpy()
            L = len(values)

            if L < self.window_size:
                continue

            for start in range(0, L - self.window_size + 1, self.step_size):
                end = start + self.window_size
                window = values[start:end]

                segments.append(window)
                labels.append(group["Label"].iloc[0])
                recording_ids.append(rid)

        return np.array(segments), np.array(labels), np.array(recording_ids)

    # --------------------------------------------------------------
    # Main Segmentation Logic
    # --------------------------------------------------------------
    def segment(self):
        self.logger.info(f"Segmenting data: window={self.window_size}, step={self.step_size}")

        try:
            # Load normalized data (already in the class)
            train = self.train_df.copy()
            val = self.val_df.copy()
            test = self.test_df.copy()

            # Identify numeric feature columns
            exclude_cols = ["Label", "RecordingID", "timestamps", "Index"]
            feature_cols = [
                c for c in train.columns
                if c not in exclude_cols and pd.api.types.is_numeric_dtype(train[c])
            ]

            if not feature_cols:
                raise ValueError("No feature columns found for segmentation.")

            self.logger.info(f"Using feature columns for segmentation: {feature_cols}")

            # Segment each split
            train_X, train_y, train_ids = self._segment_one(train, feature_cols)
            val_X, val_y, val_ids = self._segment_one(val, feature_cols)
            test_X, test_y, test_ids = self._segment_one(test, feature_cols)

            self.logger.info(f"Train windows: {len(train_X)}")
            self.logger.info(f"Val windows:   {len(val_X)}")
            self.logger.info(f"Test windows:  {len(test_X)}")

            # Save as pickle (better for ML)
            with open(self.segmented_dir / "train_windows.pkl", "wb") as f:
                pickle.dump((train_X, train_y, train_ids), f)
            with open(self.segmented_dir / "val_windows.pkl", "wb") as f:
                pickle.dump((val_X, val_y, val_ids), f)
            with open(self.segmented_dir / "test_windows.pkl", "wb") as f:
                pickle.dump((test_X, test_y, test_ids), f)

            # Save simple CSV for debugging (one row per window)
            pd.DataFrame({
                "RecordingID": train_ids,
                "Label": train_y
            }).to_csv(self.segmented_dir / "train_windows.csv", index=False)

            pd.DataFrame({
                "RecordingID": val_ids,
                "Label": val_y
            }).to_csv(self.segmented_dir / "val_windows.csv", index=False)

            pd.DataFrame({
                "RecordingID": test_ids,
                "Label": test_y
            }).to_csv(self.segmented_dir / "test_windows.csv", index=False)

            # Cache everything
            with open(self.cache_path, "wb") as f:
                pickle.dump(
                    {
                        "train": (train_X, train_y, train_ids),
                        "val": (val_X, val_y, val_ids),
                        "test": (test_X, test_y, test_ids),
                        "feature_cols": feature_cols,
                        "window_size": self.window_size,
                        "step_size": self.step_size
                    },
                    f
                )

            self.logger.info("Segmentation complete and cached successfully.")

        except Exception as e:
            self.logger.error(f"Error during segmentation: {e}")
            raise
