import os
from datetime import datetime

class BufferedLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(
            log_dir, 
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.buffer = []
        
    def log(self, message):
        self.buffer.append(f"{datetime.now().isoformat()}: {message}")
        
    def flush(self):
        if self.buffer:
            with open(self.log_file, 'a') as f:
                f.write('\n'.join(self.buffer) + '\n')
            self.buffer = []