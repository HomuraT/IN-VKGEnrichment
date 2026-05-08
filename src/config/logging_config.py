"""
VKG Data Access项目统一日志配置模块

使用Loguru库提供结构化日志记录功能，支持：
- 彩色控制台输出
- 文件日志轮转
- 错误日志单独记录
- 结构化日志格式
- 性能监控装饰器
"""

from loguru import logger
import sys
from pathlib import Path
from functools import wraps
import time
from typing import Callable, Any, Optional


def setup_logging(app_name: str = "vkg_data_access", log_level: str = "INFO", 
                  enable_file_logging: bool = True) -> None:
    """
    设置应用日志配置
    
    Args:
        app_name: 应用名称，用于日志文件命名
        log_level: 日志级别，可选: DEBUG, INFO, WARNING, ERROR, CRITICAL
        enable_file_logging: 是否启用文件日志记录
        
    Returns:
        None
    """
    # 创建日志目录
    if enable_file_logging:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
    
    # 移除默认处理器
    logger.remove()
    
    # 控制台输出（彩色）
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    if enable_file_logging:
        # 一般日志文件
        logger.add(
            log_dir / f"{app_name}_{{time:YYYY-MM-DD}}.log",
            rotation="00:00",  # 每天午夜轮转
            retention="30 days",  # 保留30天
            compression="zip",  # 压缩旧日志
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            encoding="utf-8"
        )
        
        # 错误日志单独文件
        logger.add(
            log_dir / f"{app_name}_error_{{time:YYYY-MM-DD}}.log",
            rotation="00:00",
            retention="90 days",  # 错误日志保留更久
            compression="zip",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
            encoding="utf-8"
        )


def setup_dev_logging() -> None:
    """
    开发环境日志配置：只输出到控制台，更简洁的格式
    
    Returns:
        None
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <5}</level> | <cyan>{function}</cyan> - <level>{message}</level>",
        level="DEBUG",
        colorize=True
    )


def setup_prod_logging(app_name: str = "vkg_data_access") -> None:
    """
    生产环境日志配置：结构化JSON日志
    
    Args:
        app_name: 应用名称
        
    Returns:
        None
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.remove()
    
    # 结构化JSON日志
    logger.add(
        log_dir / f"{app_name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        serialize=True,  # JSON格式
        encoding="utf-8"
    )
    
    # 错误日志
    logger.add(
        log_dir / f"{app_name}_error.log",
        level="ERROR",
        rotation="10 MB",
        retention="90 days",
        encoding="utf-8"
    )


def log_function_call(include_args: bool = False, include_result: bool = False,
                     log_level: str = "DEBUG") -> Callable:
    """
    记录函数调用的装饰器
    
    Args:
        include_args: 是否包含参数信息
        include_result: 是否包含返回值信息
        log_level: 日志级别
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            func_name = func.__name__
            
            # 记录函数开始
            if include_args:
                logger.log(log_level, f"开始执行 {func_name}，参数: args={args}, kwargs={kwargs}")
            else:
                logger.log(log_level, f"开始执行 {func_name}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # 记录成功结果
                if include_result:
                    logger.log(log_level, f"{func_name} 执行成功，耗时 {execution_time:.3f}s，返回值: {result}")
                else:
                    logger.log(log_level, f"{func_name} 执行成功，耗时 {execution_time:.3f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func_name} 执行失败，耗时 {execution_time:.3f}s，错误: {e}")
                raise
                
        return wrapper
    return decorator


def log_progress(total: int, desc: str = "处理进度") -> Callable:
    """
    记录批处理进度的装饰器
    
    Args:
        total: 总数量
        desc: 进度描述
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger.info(f"🚀 开始{desc}，总计 {total} 项")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"🎉 {desc}完成，耗时 {execution_time:.2f}s")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"❌ {desc}失败，耗时 {execution_time:.2f}s，错误: {e}")
                raise
                
        return wrapper
    return decorator


# 提供一个全局logger实例，预配置好基本设置
def get_logger(name: Optional[str] = None):
    """
    获取配置好的logger实例
    
    Args:
        name: logger名称，默认为调用模块名
        
    Returns:
        配置好的logger实例
    """
    if name:
        return logger.bind(module=name)
    else:
        return logger


# 在模块加载时进行基本配置
setup_logging()
