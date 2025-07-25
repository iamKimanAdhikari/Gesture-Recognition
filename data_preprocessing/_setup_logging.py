import logging
from pathlib import Path

class SetupLogs:
    _configured_loggers = set()
    
    def __init__(self, file_type):
        self.file_type = file_type.title()
        self.base_path = Path(__file__).parent.parent
        self.logs_path = self.base_path / "Logs"
        
    def setup_logging(self):
        logger_name = f"{self.file_type}_logger"
        
        if logger_name in SetupLogs._configured_loggers:
            return logging.getLogger(logger_name)
        
        self.logs_dir = self.logs_path / f"{self.file_type}"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
    
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
   
        file_handler = logging.FileHandler(self.logs_dir / f"{self.file_type}.log")
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        SetupLogs._configured_loggers.add(logger_name)
        
        logger.info(f"Logging setup complete for {self.file_type}")
        return logger