import logging as log
from pathlib import Path


class SetupLogs:
    def __init__(self, file_type):
        self.file_type = file_type.title()
        self.base_path = Path(__file__).parent.parent
        self.logs_path = self.base_path/"Logs"
        
    def setup_logging(self):
        self.logs_dir = self.logs_path/ f"{self.file_type}"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        log.basicConfig(
        level=log.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            log.FileHandler(self.logs_dir / f"{self.file_type}.log"),
            log.StreamHandler()
        ]
        )
        log.info("Logging Configuration Successful.")
    
def main():
    setup = SetupLogs("Check")
    setup.setup_logging()
if __name__ == "__main__":
    main()