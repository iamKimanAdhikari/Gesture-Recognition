from _get_gestures import GetGestures
from collections import defaultdict

class GetFiles(GetGestures):
    def __init__(self):
        super().__init__()
        self.gestures_dict = defaultdict(list)
        for csv in self.raw_csv_files:
            for gesture in self.sorted_gestures:
                if csv.stem.split("_")[0] in [gesture]:
                    self.gestures_dict[gesture].append(csv)

def main():
    files = GetFiles()

if __name__== "__main__":
    main()