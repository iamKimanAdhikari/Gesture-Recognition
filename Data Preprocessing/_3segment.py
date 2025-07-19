from _2clean import CleanData
from _setup_logging import SetupLogs
from torch.utils import data
import pandas as pd
import pickle

class SegmentandLabelData(CleanData):
    def __init__(self):
        super().__init__()
        setup = SetupLogs("Segment Data")
        self.logger = setup.setup_logging()
        self.segmented_dfs = {}
        self.cache_path = self.segmented_dir / "_cached_segmented_dfs.pkl"
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self.segmented_dfs = pickle.load(f)
    def segment_data(self):
        ...