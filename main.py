from data_preprocessing import JsontoCsv, Combine, CleanData, SplitData, NormalizeData
from model import TrainTCNModel

class GestureRecognitionPipeline:
    def __init__(self):
        pass
        
    def run(self):
        print("Starting gesture recognition pipeline...")
        
        # Data preprocessing
        print("Step 1: Converting JSON to CSV")
        json_converter = JsontoCsv()
        json_converter.convert()
        
        print("Step 2: Combining CSV files")
        combiner = Combine()
        
        print("Step 3: Cleaning data")
        cleaner = CleanData()
        
        print("Step 4: Splitting data")
        splitter = SplitData()

        print("Step 5: Normalizing data")
        normalizer = NormalizeData()

        # Model training
        print("Step 6: Training TCN model")
        model_trainer = TrainTCNModel()
        model_trainer.train_and_evaluate()

        print("Pipeline completed successfully!")

if __name__ == "__main__":
    pipeline = GestureRecognitionPipeline()
    pipeline.run()