import importlib
import logging
import threading

import pytest

from logger.logger import Logger


logger_module = importlib.import_module("logger")


def _remove_new_handlers(backend, initial_handlers):
    for handler in list(backend.handlers):
        if handler not in initial_handlers:
            backend.removeHandler(handler)
            handler.close()


@pytest.fixture
def reset_shared_logger(monkeypatch):
    """Reset the package singleton and clean up handlers created by the test."""
    backend = logging.getLogger("CIM")
    initial_handlers = tuple(backend.handlers)
    monkeypatch.setattr(logger_module, "_logger", None)

    yield

    _remove_new_handlers(backend, initial_handlers)


@pytest.fixture
def real_logger(monkeypatch, request):
    """Create an isolated real Logger instance for implementation tests."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_TO_FILE", "0")
    monkeypatch.setenv("LOG_INDENT_STEP", "2")

    instance = Logger(name=f"test.{request.node.name}")

    yield instance

    _remove_new_handlers(instance._logger, ())


def test_proxy_lazily_initializes_real_logger(reset_shared_logger):
    proxy = logger_module.LoggerProxy()

    assert logger_module._logger is None
    assert proxy.name == "CIM"
    assert isinstance(logger_module._logger, Logger)


def test_concurrent_callers_share_one_real_logger(reset_shared_logger):
    instances = []
    start = threading.Barrier(32)

    def get_shared_logger():
        start.wait()
        instances.append(logger_module.get_logger())

    threads = [threading.Thread(target=get_shared_logger) for _ in range(32)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(instances) == 32
    assert all(isinstance(instance, Logger) for instance in instances)
    assert len({id(instance) for instance in instances}) == 1


def test_logger_outputs_module_tag_and_indent(real_logger, capfd):
    real_logger._push_module_tag("compiler")
    try:
        with real_logger.indent(step=4):
            real_logger.info("compiling")
    finally:
        real_logger._pop_module_tag()

    output = capfd.readouterr().err

    assert "[INFO]" in output
    assert "[compiler]:     compiling" in output


def test_module_tag_is_thread_local(real_logger):
    worker_tags = {}
    real_logger._push_module_tag("main")

    def worker():
        worker_tags["before"] = real_logger._get_module_tag()
        real_logger._push_module_tag("worker")
        try:
            worker_tags["during"] = real_logger._get_module_tag()
        finally:
            real_logger._pop_module_tag()
        worker_tags["after"] = real_logger._get_module_tag()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    try:
        assert worker_tags == {
            "before": "-",
            "during": "worker",
            "after": "-",
        }
        assert real_logger._get_module_tag() == "main"
    finally:
        real_logger._pop_module_tag()
