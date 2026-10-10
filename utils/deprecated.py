"""
标记已过期的 API

example function:
    @deprecated(
        since="1.5",
        remove="2.0",
        replacement="new_api",
        reason="API redesigned"
    )
    def old_api():
        pass

example class:
    @deprecated(
        since="2.0",
        replacement="NewModel"
    )
    class OldModel:

        def __init__(self):
            pass
"""

import functools
import inspect
import warnings
from enum import Enum


class APIStatus(str, Enum):
    EXPERIMENTAL = "experimental"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    REMOVED = "removed"


class DeprecatedWarning(DeprecationWarning):
    """Custom warning category for deprecated APIs."""

    pass


class DeprecatedError(RuntimeError):
    """Raised when deprecated API is configured as removed."""

    pass


class ExperimentalWarning(UserWarning):
    """Warning for experimental APIs."""

    pass


STATUS_LEVEL = {
    APIStatus.EXPERIMENTAL.value: {"action": "warning", "category": ExperimentalWarning},
    APIStatus.STABLE.value: {"action": "none", "category": None},
    APIStatus.DEPRECATED.value: {"action": "warning", "category": DeprecatedWarning},
    APIStatus.REMOVED.value: {"action": "error", "category": DeprecatedError},
}


def _build_message(obj, metadata):
    name = getattr(obj, "__name__", str(obj))
    status = metadata.get("status")
    msg = f"{name} API status: {status}."

    if metadata.get("since"):
        msg += f" Since {metadata['since']}."
    if metadata.get("remove"):
        msg += f" Remove in {metadata['remove']}."
    if metadata.get("replacement"):
        msg += f" Use {metadata['replacement']} instead."
    if metadata.get("reason"):
        msg += f" Reason: {metadata['reason']}."

    return msg


def _attach_metadata(obj, metadata):
    try:
        obj.__api_status__ = metadata
    except Exception:
        pass
    return obj


def _handle_status(message, metadata, category=None):
    status = metadata["status"]
    config = STATUS_LEVEL.get(status, {"action": "none"})

    action = config["action"]
    if action == "error":
        raise DeprecatedError(message)
    if action == "warning":
        warnings.warn(message, category or config["category"], stacklevel=3)


def api_status(*, status="stable", since=None, remove=None, replacement=None, reason=None, category=None):
    if status not in [s.value for s in APIStatus]:
        raise ValueError(f"Unsupported status: {status}")

    metadata = {
        "status": status,
        "since": since,
        "remove": remove,
        "replacement": replacement,
        "reason": reason,
    }

    def decorator(obj):
        if isinstance(obj, staticmethod):
            func = obj.__func__
            wrapped = _wrap_function(func, metadata, category)
            return staticmethod(_attach_metadata(wrapped, metadata))
        elif isinstance(obj, classmethod):
            func = obj.__func__
            wrapped = _wrap_function(func, metadata, category)
            return classmethod(_attach_metadata(wrapped, metadata))
        elif inspect.isclass(obj):
            wrapped = _wrap_class(obj, metadata, category)
            return _attach_metadata(wrapped, metadata)
        else:
            wrapped = _wrap_function(obj, metadata, category)
            return _attach_metadata(wrapped, metadata)

    return decorator


def deprecated(**kwargs):
    """
    Mark an API as deprecated.
    """
    kwargs["status"] = APIStatus.DEPRECATED.value
    return api_status(**kwargs)


def _wrap_function(func, metadata, category):
    message = _build_message(func, metadata)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _handle_status(message, metadata, category)
        return func(*args, **kwargs)

    return wrapper


def _wrap_class(cls, metadata, category):
    message = _build_message(cls, metadata)

    original_init = cls.__init__

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        _handle_status(message, metadata, category)
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init

    return cls


def get_api_info(obj):
    """
    Get API lifecycle metadata.
    """
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__

    return getattr(obj, "__api_status__", None)
