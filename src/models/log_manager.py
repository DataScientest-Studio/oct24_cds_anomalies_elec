import logging
from logging.handlers import RotatingFileHandler
import sys
import os

def setup_logger(log_dir="./logs", log_file="mon_execution.log", max_bytes=10*1024*1024, backup_count=5):
    """
    Initialise un logger thread-safe avec rotation automatique des fichiers de log.
    Capture aussi stdout et stderr (print et erreurs).
    """

    # Création du répertoire si nécessaire
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # Création du logger
    logger = logging.getLogger("mon_logger")
    logger.setLevel(logging.INFO)

    # Evite les handlers en double si plusieurs appels
    if not logger.handlers:
        handler = RotatingFileHandler(log_path, mode='a',
                                       maxBytes=max_bytes,
                                       backupCount=backup_count)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Redirection stdout/stderr
        class StreamToLogger(object):
            def __init__(self, logger, log_level=logging.INFO):
                self.logger = logger
                self.log_level = log_level
                self.linebuf = ''
            def write(self, buf):
                for line in buf.rstrip().splitlines():
                    self.logger.log(self.log_level, line.rstrip())
            def flush(self):
                pass

        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)

    return logger
