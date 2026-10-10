"""YAML 字段校验; 每个函数接收原始值和字段路径, 返回规范化后的值或抛出带路径的错误"""

from __future__ import annotations

import ipaddress
import operator
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from .models import ExtractionConfig


class ProjectConfigError(ValueError):
    """项目配置无效"""


def mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectConfigError(f"{field} must be a mapping")
    return value


def string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectConfigError(f"{field} must be a non-empty string")
    return value.strip()


def integer(value: Any, field: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ProjectConfigError(f"{field} must be an integer")
    try:
        value = operator.index(value)
    except TypeError:
        raise ProjectConfigError(f"{field} must be an integer") from None
    if minimum is not None and value < minimum:
        raise ProjectConfigError(f"{field} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ProjectConfigError(f"{field} must be at most {maximum}")
    return value


def boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ProjectConfigError(f"{field} must be true or false")
    return value


def port(value: Any, field: str) -> int:
    return integer(value, field, minimum=1, maximum=65535)


def ip_address(value: Any, field: str) -> str:
    value = string(value, field)
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        raise ProjectConfigError(f"{field} must be an IPv4/IPv6 address") from None


def ip_network(value: Any, field: str) -> str:
    """单个 IP 或严格 CIDR"""
    value = string(value, field)
    try:
        ipaddress.ip_network(value, strict=True)
    except ValueError:
        raise ProjectConfigError(f"{field} must be an IP address or strict CIDR network") from None
    return value


def project_path(value: Any, field: str, *, root: Path) -> Path:
    path = Path(string(value, field)).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def component(value: Any, field: str) -> str:
    """文件名片段: 芯片名、版本号、包前缀"""
    value = string(value, field)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", value):
        raise ProjectConfigError(f"{field} must be a valid filename component")
    return value


def package_suffix(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"(?:_[A-Za-z0-9][A-Za-z0-9._+-]*)?", value):
        raise ProjectConfigError(f"{field} must be empty or a filename suffix such as '_tmp'")
    return value


def download_auth(value: Any, field: str) -> str:
    value = string(value, field)
    if value not in {"basic", "bearer"}:
        raise ProjectConfigError(f"{field} must be 'basic' or 'bearer'")
    return value


def _relative_dir(value: str, field: str, *, single_component: bool) -> str:
    path = PurePosixPath(value)
    invalid = (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or any(char in value for char in "\\*?[]\x00\r\n")
        or (single_component and len(path.parts) != 1)
    )
    if invalid:
        kind = (
            "one relative directory name" if single_component else "a relative directory path without '..' or wildcards"
        )
        raise ProjectConfigError(f"{field} must be {kind}")
    return path.as_posix()


def extraction(value: Any, field: str) -> ExtractionConfig:
    raw = mapping(value, field)
    unexpected = sorted(set(raw) - {"wrapper_dir", "exclude_dirs"})
    if unexpected:
        raise ProjectConfigError(f"{field}.{unexpected[0]} is not a supported setting")

    wrapper_dir = raw.get("wrapper_dir")
    if wrapper_dir is not None:
        wrapper_dir = _relative_dir(
            string(wrapper_dir, f"{field}.wrapper_dir"), f"{field}.wrapper_dir", single_component=True
        )

    excluded = raw.get("exclude_dirs", [])
    if not isinstance(excluded, list):
        raise ProjectConfigError(f"{field}.exclude_dirs must be a list")
    paths = tuple(
        _relative_dir(
            string(item, f"{field}.exclude_dirs[{index}]"), f"{field}.exclude_dirs[{index}]", single_component=False
        )
        for index, item in enumerate(excluded)
    )
    return ExtractionConfig(wrapper_dir=wrapper_dir, exclude_dirs=paths)


def base_url(value: Any, field: str) -> str:
    value = string(value, field).rstrip("/")
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and not parsed.query
            and not parsed.fragment
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        valid = False
    if not valid:
        raise ProjectConfigError(f"{field} must be an HTTP(S) base URL without credentials, query or fragment")
    return value


def gitlab_project_url(value: Any, field: str) -> str:
    """把 /api/v4/projects/<group/sub/project> 的项目路径编码成一个段; 兼容已编码路径和数字 ID"""
    parsed = urlsplit(base_url(value, field))
    api_prefix = "/api/v4/projects/"
    if not parsed.path.startswith(api_prefix):
        raise ProjectConfigError(
            f"{field} must be a GitLab API project URL: https://host/api/v4/projects/<project path or ID>"
        )
    project = parsed.path[len(api_prefix) :]
    return parsed._replace(path=f"{api_prefix}{quote(unquote(project), safe='')}").geturl()
