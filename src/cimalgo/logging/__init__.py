"""
进程级共享 logger

用法::

    from cimalgo.logging import get_logger

    log = get_logger(__name__)
    log.info("开始部署 %s", name)

    with log.indent():
        log.debug("子步骤")

    @log.trace()
    def compile_model(...): ...

所有模块通过 get_logger(__name__) 取得各自命名的 logger, 它们共享同一套 handler 和缩进状态。
配置从环境变量读取: CIM_LOG_LEVEL (默认 INFO)、CIM_LOG_FILE (设置后同时写文件)、CIM_LOG_COLOR (0 关闭颜色)。
pytest 运行时 handler 输出到 stderr, 由 pytest 的 capture 机制收集, 失败时随报告展示。
"""

from .logger import CimLogger, configure_logging, get_logger

__all__ = ["CimLogger", "configure_logging", "get_logger"]
