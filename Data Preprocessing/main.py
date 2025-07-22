from _0jsontocsv import JsontoCsv
from _1combine import Combine
from _2clean import CleanData
from _3split import SplitData

class DataPreprocessing:
    def __init__(self):
        pass
        
    def run_all(self):
        # Step 1: Convert JSON to CSV
        json_converter = JsontoCsv()
        json_converter.convert()
        
        # Step 2: Combine CSVs
        combiner = Combine()
        
        # Step 3: Clean data
        cleaner = CleanData()
        
        # Step 4: Split data
        splitter = SplitData()

def main():
    datapreprocessing = DataPreprocessing()
    datapreprocessing.run_all()

if __name__ == "__main__":
    main()