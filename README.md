# CIMAlgo_Auto

BW01 / BW01D CIM 芯片的自动化测试框架，基于 pytest。负责按需下载解压 SDK/工具链、部署测试环境变量、通过 SSH 操作测试设备、解析测试 case、分析并绘制测试结果。

## 环境准备

- Python 3.12
- 安装依赖：

  ```bash
  pip install -r requirements.txt

  # PyPI 访问慢时换成清华镜像
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

- 准备 `.netrc`（项目根目录，权限 `600`），为下载 SDK/工具链提供凭据。认证方式
  由 `cim_config.yaml` 里每个 SDK/工具链各自的 `auth` 字段决定：

  ```
  machine <host>
  login <username>
  password <password-or-token>
  ```

## 配置 `cim_config.yaml`

- `active_sdk`：默认启用的 SDK，对应 `sdks` 下的键；也可以用 `--sdk` 或环境变量
  `BW_SDK` 覆盖，优先级为 `--sdk` > `BW_SDK` > `active_sdk`。
- `sdks.<name>`：`chip`（芯片标识，决定 `SetEnv`/`DeviceOps` 选哪个实现）、
  `version`、`sdk_prefix`、下载 `url`、`auth`、所引用的 `toolchain`。
- `toolchains.<name>`：被多个 SDK 共用的工具链定义，包含 `map_tools` 和
  `compiler_tools` 两个包的 `prefix`/`suffix`/`extract` 配置。
- `url`/`auth` 可以省略（`url: null`），此时按本地已安装处理，不会触发下载。
- 完整字段语义见 [cim_config.py](cim_config.py) 的 `load_project_config` 文档字符串。

## 运行测试

```bash
# 首次运行，或需要重新下载/安装 SDK、工具链、同步 wheel
pytest --download

# 日常运行（已安装好环境）
pytest

# 指定 SDK（覆盖 active_sdk）
pytest --sdk bw01d_v2.3.4

# 并行执行（pytest-xdist），--download 只会在主进程执行一次
pytest --download -n auto
```

## 目录结构

```
.
|-- cim_config.py            配置数据模型 + load_project_config 组装逻辑
|-- cim_config_validators.py YAML 字段校验，被 cim_config.py 复用
|-- cim_constants.py         项目级常量
|-- conftest.py              pytest hooks：加载配置、部署环境、下载安装编排
|-- analyzer/                测试结果分析与绘图
|-- device_ctrl/             设备操作契约（DeviceOps）与具体驱动实现
|-- logger/                  进程级共享 logger
|-- tests/                   与上述模块一一对应的单元测试
`-- utils/                   环境部署（SetEnv）、SSH 连接、设备锁等工具
```

## 扩展

### 新增环境配置

`SetEnv`（`utils/set_env.py`）按 `config.sdk.chip` 选择实现，调用方（`conftest.py`）
不需要改动。新增环境配置只需要定义一个带 `chip=` 的子类：

```python
class NewChipSetEnv(SetEnv, chip="new_chip"):
    def chip_environment(self) -> Mapping[str, str]:
        ...  # 这个芯片特有的环境变量

    # 需要下载解压、补执行权限、同步 wheel、调整导入路径时，
    # 按需覆盖 packages() / executable_targets() / sync_wheels() / add_import_paths()
```

### 新增 SDK API

`DeviceOps`（`device_ctrl/sdk/ops.py`）按 `DeviceTarget.driver` 选择实现：

```python
class NewSDKOps(DeviceOps, driver="new_driver"):
    def enable_power_channels(self) -> None: ...
    ...  # 实现 DeviceOps 的其余抽象方法
```

```python
DeviceOps.for_target(DeviceTarget(..., driver="new_driver"))
```

## 代码风格

请先安装 [ruff](https://docs.astral.sh/ruff/)

```bash
pip install ruff

# PyPI 访问慢时换成清华镜像
pip install ruff -i https://pypi.tuna.tsinghua.edu.cn/simple
```

用 ruff 做 lint 和格式化，配置见 `pyproject.toml`：

```bash
ruff check .                  # 只读，报告 lint 问题
ruff check --fix .            # 自动修复可安全修复的 lint 问题（不含格式化）
ruff check path/to/file.py    # 只检查单个文件/目录
ruff format .                 # 按 [tool.ruff.format] 重排代码，写回文件
ruff format --check .         # 只检查格式是否已符合要求，不写回文件
ruff format --diff .          # 显示格式化前后的差异，不写回文件
```
