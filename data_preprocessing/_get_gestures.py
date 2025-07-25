from pathlib import Path
from ._setup_directories import SetupDirectories  

class GetGestures(SetupDirectories):
    def __init__(self):
        super().__init__()
        gestures_list = []
        for c in self.raw_csv_files:
            gestures_list.append(c.stem.split("_")[0])
        self.sorted_gestures = sorted(set(gestures_list))


def main(): 
    GetGestures()

if __name__ == "__main__":
    main()