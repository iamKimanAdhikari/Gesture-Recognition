from pathlib import Path

def get_gestures(csv_dir = (Path(__file__).parent.parent / "Data" / "Raw_Data" / "csvRaw")):
    csv_files = list(csv_dir.glob("*.csv"))
    gestures_list = []
    for c in csv_files:
        gestures_list.append(c.stem.split("_")[0])
    sorted_gestures = sorted(set(gestures_list))
    return sorted_gestures

def main():
    print(get_gestures()) 

if __name__ == "__main__":
    main()