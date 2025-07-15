from _get_gestures import get_gestures
from pathlib import Path
from collections import defaultdict

def get_files(csv_dir = (Path(__file__).parent.parent / "Data" / "Raw_Data" / "csvRaw")):
    gestures = get_gestures()
    gestures_dict = defaultdict(list)
    csv_files = sorted(
        csv_dir.glob("*.csv"),
        key=lambda p: int(p.stem.split("_s")[1].split("_")[0]) if "_s" in p.stem else 0
    )
    
    for csv in csv_files:
        for gesture in gestures:
            if csv.stem.split("_")[0] in [gesture]:
                gestures_dict[gesture].append(csv)
    return gestures_dict

if __name__== "__main__":
    get_files()