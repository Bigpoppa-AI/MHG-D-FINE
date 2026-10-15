import sys, logging
import torch.distributed as dist
    
class ColorFormatter(logging.Formatter): 
    RESET = "\033[0m"
    COLORS = {
        "TIME": "\033[92m",       # 绿色
        "LOCATION": "\033[93m",   # 黄色    
        "INFO": "\033[37m",       # 白色    
        "DEBUG": "\033[36m",      # 青色   
        "WARNING": "\033[33m",    # 黄色
        "ERROR": "\033[31m",      # 红色
        "CRITICAL": "\033[1;31m", # 红色加粗    
    } 

    def format(self, record):
        time_str = f"{self.COLORS['TIME']}{self.formatTime(record, self.datefmt)}{self.RESET}"
        location = f"{self.COLORS['LOCATION']}[{record.filename}:{record.funcName}:{record.lineno}]{self.RESET}"
        level_color = self.COLORS.get(record.levelname, self.RESET)    
        levelname = f"{level_color}{record.levelname}{self.RESET}"  
        message = f"{level_color}{record.getMessage()}{self.RESET}"
   
        return f"{time_str} {location} {levelname}: {message}"

def get_logger(name=None, level=logging.INFO):
    """    
    创建并返回一个 logger     
    :param name: logger 名字，一般用 __name__   
    :param log_file: 可选，日志写入文件
    :param level: 日志级别
    :return: logger 对象
    """
    logger = logging.getLogger(name)    
    logger.setLevel(level)     
    logger.propagate = False     
     
    is_main_process = True
    if dist.is_available() and dist.is_initialized():  
        is_main_process = dist.get_rank() == 0 
     
    # 避免重复添加 Handler
    if not logger.handlers and is_main_process:    
        # 控制台输出   
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)   
        formatter = ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter) 
        logger.addHandler(ch)
     
    return logger 
     
def test_logger(): 
    logger = get_logger(__name__)  
    logger.info("info")
    logger.warning("info")
    logger.error("info") 
