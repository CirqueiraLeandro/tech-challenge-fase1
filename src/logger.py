import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name, config=None, log_level=None):
    """
    Setup centralized logger com console e file handlers
    
    Args:
        name: Logger name
        config: Configuration object (optional, defaults to get_config())
        log_level: Log level override (optional)
    """
    from src.config import get_config
    
    if config is None:
        config = get_config()
    
    logger = logging.getLogger(name)
    
    # Evitar adicionar múltiplos handlers
    if logger.handlers:
        return logger
    
    log_level_str = log_level or config.LOG_LEVEL
    log_level_int = getattr(logging, log_level_str)
    logger.setLevel(log_level_int)
    
    # Criar diretório de logs
    config.LOGS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Console Handler — UTF-8 para suportar caracteres especiais no Windows
    import sys
    stream = sys.stdout
    try:
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    except Exception:
        pass
    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(log_level_int)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File Handler
    file_handler = RotatingFileHandler(config.LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(log_level_int)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Adicionar handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
