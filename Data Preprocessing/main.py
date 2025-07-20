from _0jsontocsv import JsontoCsv
from _1combine import Combine
from _2clean import CleanData

class DataPreprocessing:
    def __init__(self):
        # Only initialize JSON converter first
        self.jsontocsv = JsontoCsv()
        
    def run_all(self):
        # Step 1: Convert JSON to CSV
        self.jsontocsv.convert()
        
        # Step 2: Initialize and run combiner AFTER CSV files exist
        self.combiner = Combine()
        self.combiner.combine()
        
        # Step 3: Initialize and run cleaner with combined data
        self.cleaner = CleanData(combined_data=self.combiner.all_dfs)
        self.cleaner.remove_outliers()

def main():
    datapreprocessing = DataPreprocessing()
    datapreprocessing.run_all()

if __name__ == "__main__":
    main()