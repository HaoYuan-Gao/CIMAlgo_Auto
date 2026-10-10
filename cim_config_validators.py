"""cim_config.py 使用的 YAML 字段校验: 路径、URL、通用配置片段"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

from cim_constants import PROJECT_ROOT
from utils.type_checks import require_mapping, require_string


class ProjectConfigError(ValueError):
    """Raised when the global project configuration is invalid."""


config_mapping = partial(require_mapping, error_type=ProjectConfigError)
config_string = partial(require_string, error_type=ProjectConfigError)


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    wrapper_dir: str | None = None
    exclude_dirs: tuple[str, ...] = ()


def project_path(value, field: str) -> Path:
    """Resolve paths against the project root, including custom YAML files"""
    path = Path(config_string(value, field)).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def component(value, field: str) -> str:
    """Validate a chip, version or package prefix used in paths and URLs"""
    value = config_string(value, field)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", value):
        raise ProjectConfigError(f"{field} must be a valid filename component")
    return value


def package_suffix(value, field: str) -> str:
    """校验追加到工具包版本和 Release tag 后面的可选后缀。"""
    if not isinstance(value, str) or not re.fullmatch(r"(?:_[A-Za-z0-9][A-Za-z0-9._+-]*)?", value):
        raise ProjectConfigError(f"{field} must be empty or a filename suffix such as '_tmp'")
    return value


def download_auth(value, field: str) -> str:
    """下载认证方式由配置指定, 与仓库类型无关"""
    value = config_string(value, field)
    if value not in {"basic", "bearer"}:
        raise ProjectConfigError(f"{field} must be 'basic' or 'bearer'")
    return value


def extraction_config(value, field: str) -> ExtractionConfig:
    """校验包装目录和相对于内容根目录的排除路径。"""
    raw = config_mapping(value, field)
    unexpected = set(raw) - {"wrapper_dir", "exclude_dirs"}
    if unexpected:
        name = sorted(unexpected)[0]
        raise ProjectConfigError(f"{field}.{name} is not a supported setting")

    wrapper_dir = raw.get("wrapper_dir")
    if wrapper_dir is not None:
        wrapper_dir = config_string(wrapper_dir, f"{field}.wrapper_dir")
        wrapper_path = PurePosixPath(wrapper_dir)
        if (
            wrapper_path.is_absolute()
            or len(wrapper_path.parts) != 1
            or ".." in wrapper_path.parts
            or any(char in wrapper_dir for char in "\\*?[]\x00\r\n")
        ):
            raise ProjectConfigError(f"{field}.wrapper_dir must be one relative directory name")
        wrapper_dir = wrapper_path.parts[0]

    excluded = raw.get("exclude_dirs", [])
    if not isinstance(excluded, list):
        raise ProjectConfigError(f"{field}.exclude_dirs must be a list")
    paths = []
    for index, value in enumerate(excluded):
        path_field = f"{field}.exclude_dirs[{index}]"
        value = config_string(value, path_field)
        path = PurePosixPath(value)
        if (
            path.is_absolute()  # path 表示压缩包内要排除的目录，约定它使用相对路径
            or not path.parts
            or ".." in path.parts
            or any(char in value for char in "\\*?[]\x00\r\n")
        ):
            raise ProjectConfigError(f"{path_field} must be a relative directory path without '..' or wildcards")
        paths.append(path.as_posix())
    return ExtractionConfig(wrapper_dir=wrapper_dir, exclude_dirs=tuple(paths))


def base_url(value, field: str) -> str:
    value = config_string(value, field).rstrip("/")
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not parsed.query  # url + ?query 给服务器传参数
            and not parsed.fragment  # url + #fragment  给客户端定位内容
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        valid = False
    if not valid:
        raise ProjectConfigError(f"{field} must be an HTTP(S) base URL without credentials, query or fragment")
    return value


def gitlab_project_url(value, field: str) -> str:
    """编码 API 项目地址中的完整项目路径, 同时兼容已编码路径和数字 ID"""
    parsed = urlsplit(base_url(value, field))
    api_prefix = "/api/v4/projects/"
    if not parsed.path.startswith(api_prefix):
        raise ProjectConfigError(
            f"{field} must be a GitLab API project URL: https://host/api/v4/projects/<project path or ID>"
        )
    project = parsed.path[len(api_prefix) :]
    # 先解码再编码, 避免配置里已有的 %2F 被重复编码为 %252F
    path = f"{api_prefix}{quote(unquote(project), safe='')}"
    return parsed._replace(path=path).geturl()
