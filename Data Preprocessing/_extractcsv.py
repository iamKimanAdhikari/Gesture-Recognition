from _3split import SplitData
import pandas as pd

class ExtractCsv:
    def __init__(self):
        # Initialize the full pipeline to access all data
        self.pipeline = SplitData()
        
    def show_data(self, stage="cleaned"):
        if stage == "raw":
            print("Raw CSV files:")
            for file in self.pipeline.raw_csv_files:
                print(f"- {file.name}")
                
        elif stage == "combined":
            print("\nCombined data sample:")
            print(self.pipeline.combined_df.head())
            
        elif stage == "cleaned":
            print("\nCleaned data sample:")
            print(self.pipeline.cleaned_df.head())
            
        elif stage == "split":
            print("\nSplit data:")
            print(f"Training samples: {len(self.pipeline.train_df)}")
            print(f"Validation samples: {len(self.pipeline.val_df)}")
            print(f"Testing samples: {len(self.pipeline.test_df)}")
            
        else:
            print(f"Invalid stage: {stage}. Use 'raw', 'combined', 'cleaned', or 'split'")

if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "cleaned"
    extractor = ExtractCsv()
    extractor.show_data(stage)