# cimalgo

BW01 / BW01D CIM 芯片自动化测试框架。基于 pytest，用 YAML 描述测试 case，框架负责
准备环境、连接设备、编译模型、烧录、推理、分析结果并判定阈值。

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

需要 Python 3.12+。开发机 Linux 或 macOS；CI 和设备侧为 Linux。它会把当前项目安装到所选的 Python 环境中，让该环境可以导入 `cimalgo` 并使用项目提供的命令。同时，相关依赖也会安装到该环境中。

- `.`：当前目录，pip 从这里读取 `pyproject.toml` 中的项目名称、依赖和打包配置。
- `-e`：可编辑安装，环境中的包关联当前目录的源码，而不是使用一份独立复制的源码。
- `.[dev]`：在项目基础依赖之外，额外安装 `dev` 中的 pytest、Ruff、mypy 等工具。

`pyproject.toml` 中的配置声明了两个命令行入口：

```toml
[project.scripts]
cim-deploy-sdk = "cimalgo.remote.sdk_deploy:main"
cim-device-lock = "cimalgo.remote.device_lock:main"
```

激活该环境后，可以查看命令帮助：

```bash
cim-deploy-sdk --help
cim-device-lock --help
```

## 快速开始
1. 准备 `.netrc`（项目根目录，权限 600）。SDK / 工具链下载凭据和**每台设备的 SSH 账号密码**都放这里，
   设备按 IP 匹配：

   ```
   machine nexus.witmem.ad
   login <user>
   password <password>
   machine git.witmem.ad
   login oauth2
   password <token>

   machine 10.27.17.45
   login root
   password <evb-01 的密码>
   machine 10.27.17.46
   login tester
   password <evb-02 的密码>
   ```

   `.netrc` 里有的设备用它的账号密码；没有的回退到 `cim_config.yaml` 里的 `ssh_username` 加 `--device-password` / 环境变量 `CIM_DEVICE_PASSWORD`。
   如果在 `cim_config.yaml` 指定 `netrc:`，下载和 SSH 登录都使用该路径；相对路径以配置文件所在目录为根。

2. 在 `cim_config.yaml` 的 `devices:` 下登记测试设备（IP、SSH 端口、电源控制入口等；不放密码）。
   gRPC 端口选择板上哪颗芯片，属于用例参数，写在 case 的 `device.grpc_port` 或 `@pytest.mark.device(grpc_port=...)`。
   `evb_port` 和 `bmc_ip` 分别是不同芯片的电源控制入口，一台设备只能配置其中一个。

3. 首次运行，下载并安装 SDK 和工具链：

   ```bash
   pytest --download tests/unit
   ```

4. 跑硬件 case：

   ```bash
   pytest tests/hardware -m hardware --cases cases

   # 多台设备各开一个进程, 互不影响
   pytest tests/hardware -m hardware --cases cases --device evb-01 &
   pytest tests/hardware -m hardware --cases cases --device evb-02 &
   ```

   不加 `-m hardware` 时，所有带 `hardware` 标记的测试会被跳过，`pytest` 默认只跑单元测试。如需生成供 CI 系统读取的测试报告，可以运行:
   ```bash
   pytest tests/hardware -m hardware --cases cases --device evb-02 --junitxml=junit.xml
   ```
   `junit.xml` 是 JUnit 格式的 XML 测试结果，记录用例的通过、失败或跳过状态、耗时和失败原因。


## 写一个 case

一个 YAML 文件一个 case，放在 `cases/` 下（可以有子目录）：

```yaml
case: dense_128x256_fake      # 名字，也是 pytest 参数 ID
tags: [smoke]

device:
  name: evb-01                # cim_config.yaml devices 里的键
  grpc_port: 60000            # 选板上哪颗芯片 (site); 省略为 60000
  power_cycle: true
  hard_blocks: [0, 1, 2, 3]
  select_layers: [0]

model:
  kind: fake                  # fake: 按 layers 生成; external: 给 path
  seed: 7
  layers:
    - {name: model.layers.0.dense_L2_0_L0_0_node_0, batch: 32, in_features: 128, out_features: 256}

infer:
  dummy: 2
  mode: {repeat: 1}

thresholds:                   # 指标名见 LayerAnalyzer.summary()
  cs_overall: ">= 0.995"
  error_std_max: "<= 2.0"
```

`tests/hardware/test_pipeline.py` 是按 YAML 参数化的硬件测试函数；差异全在 YAML 里。它的每一步都是
`cimalgo.pipeline` 里的普通函数：

| 函数 | 作用 | 返回 |
|---|---|---|
| `compile_model(toolchain, model_dir, compile_dir, *, oc_comp=False, map_options=None)` | 节点配置 → 编译 → 映射打包 | `CompiledModel`（`jobs` 是与 layer 绑定的 `(L2, 压缩包)` 列表） |
| `program_model(device_ops, compiled, output_dir, *, k=1, d=1)` | 按编译产物绑定的 L2 烧录 | SDK 响应 |
| `InferSession(device_ops, result_dir)` + `infer_model(model_dir, ...)` | 推理 | `list[NodeResult]` |
| `analyze_results(results, plot_dir, *, thresholds=())` | 逐节点绘图、算指标、判阈值；不抛异常 | `AnalysisReport`（`passed` / `failures` / `flat_metrics()`） |

编译和映射保留 `Die_<die>_L2_<id>` 中的 L2 编号：逻辑 L2 和物理 L2 永远同号，模型 layer 名字里的 `L2_<n>_L0_<n>_node_<n>` 就是硬件上真正烧录、推理用的地址，不需要额外的映射关系追踪。每个 Map 的映射参数（`stage`、冗余信息）用 `MapOptions` 按 **L2 编号**配置。需要注意的是, <mark>如果 l2 + archive 是随意的组合, 烧录不会报错, 但是 infer 时会出错</mark>。
语义：

```python
from cimalgo.pipeline import MapOptions

# 按 L2 分别配置; 没列出的 L2 用默认参数
compile_model(
    ...,
    map_options={
        2: MapOptions(stage=1),
        5: MapOptions(stage=4, redundancy_info=Path("r5.bin")),
    },
)
```

键必须是已存在的 L2 编号。`oc_comp=True` 会在每个 `compile_dir/output/Die_…` 目录内执行后补偿编译。显式指定的 `--cases` 目录不存在、不是目录或没有匹配的 YAML 文件时，会报配置错误, <mark>未特殊指定测试数据路径, 则默认位于 `--cases` 目录</mark>。

### 流程不固定时：写 Python 测试

只测某个功能、循环推理、中途改设备状态再推理——这些不适合 YAML 的，直接写 pytest 测试。用 `@pytest.mark.device` 声明设备和初始化参数，`@pytest.mark.model` 声明模型来源（字段和 YAML 里一样），其余 fixture 和上面的步骤函数完全复用。`tests/hardware/test_examples.py` 有三个完整例子：

```python
DENSE = {
    "name": "model.layers.0.dense_L2_0_L0_0_node_0",
    "batch": 32,
    "in_features": 128,
    "out_features": 256,
}


@pytest.mark.hardware
@pytest.mark.device("evb-01", hard_blocks=[0, 1, 2, 3], select_layers=[0])
def test_firmware_matches_sdk(device_ops, project_config):
    # 只验证一个功能点, 不编译不烧录
    assert device_ops.get_firmware_version() == project_config.sdk.version


@pytest.mark.hardware
@pytest.mark.device("evb-01", hard_blocks=[0, 1, 2, 3], select_layers=[0])
@pytest.mark.model("fake", seed=7, layers=[DENSE])
def test_inference_is_repeatable(device_ops, project_config, model_dir, case_workdir):
    compiled = compile_model(project_config.toolchain, model_dir, case_workdir / "compile")
    program_model(device_ops, compiled, case_workdir / "program")
    with InferSession(device_ops, case_workdir / "results") as session:  # 必须在 program 之后
        runs = [session.infer_model(model_dir, prefix=f"run{i}") for i in range(10)]
    ...


@pytest.mark.hardware
@pytest.mark.device("evb-01", hard_blocks=[0, 1, 2, 3], select_layers=[0], power_cycle=False)
@pytest.mark.model("fake", seed=7, layers=[DENSE])
def test_tsv_sweep(device_ops, project_config, model_dir, case_workdir):
    compiled = compile_model(project_config.toolchain, model_dir, case_workdir / "compile")
    for tsv in (0, 64, 128, 255):
        device_ops.select_layer(0, tsv_sel0=tsv)  # 手动改状态, 设备池会自动标脏
        program_model(device_ops, compiled, case_workdir / f"tsv_{tsv}" / "program")
        ...
```

YAML case 和 Python 测试混在一起收集、排序、共用设备池；两种写法没有能力差异，只是声明方式不同。

### 设备初始化的复用

所有硬件测试（YAML 和 Python）在收集后按 `(设备, gRPC 端口, 固件版本, hard_blocks, select_layers, tsv)` 排序，让要求相同芯片状态的测试相邻。会话级的设备池为每台设备保持一条 SSH 连接、一把锁和一个 `DeviceOps`，并记住上一次初始化到的状态：

- 下一个测试要求的状态相同 → 跳过初始化（日志里会打一行"跳过"）
- 固件版本、`hard_blocks`、`select_layers`、`tsv_sel0/1` 任一不同 → 完整重新初始化
- `grpc_port` 不同 → 换的是另一颗芯片：关闭 gRPC 通道、重连新端口并完整重新初始化（SSH 连接和锁不变）
- 上一个测试失败（setup 或测试体）→ 该设备状态作废，下一个测试强制重新初始化
- `update_sdk: true` 部署 SDK 之后 → 同上
- 测试体里直接调用了 `select_layer` / `set_phy_l2` / `switch_firmware_version` / `power_cycle` → 同上，
  `DeviceOps` 自己会把状态标为未知，不需要手动处理

`power_cycle` 不参与比较，它只决定这次初始化前是否断电。

## 可用的 fixture

| fixture | scope | 内容 |
|---|---|---|
| `project_config` | session | 解析后的 `cim_config.yaml` |
| `ssh_connector` | session | 带首次信任白名单的 SSH 连接工厂 |
| `workdir` | session | `--workdir` 目录（默认 `workspace/`） |
| `case` | function | 当前 YAML case（需要 `--cases`） |
| `case_workdir` | function | `workdir/<case 名>` |
| `model_spec` | function | 当前测试的模型来源（来自 YAML `model:` 或 `@pytest.mark.model`） |
| `model_dir` | function | fake 则生成，external 则校验存在 |
| `device_spec` | function | 当前测试的设备名和初始化参数（来自 YAML `device:` 或 `@pytest.mark.device`） |
| `device_ops` | function | 已初始化到 `device_spec` 所要求状态的设备操作对象；连接由设备池跨测试持有 |
| `locked_device` | function | 当前设备的 SSH 连接（已持锁），需要直接跑远程命令时用 |
| `device_pool` | session | 设备池本身；一般不直接用 |

`InferSession` 不是 fixture：<mark>它的 `init` 依赖芯片上已烧录的程序</mark>，要在测试体内 `program()` 之后用 `with InferSession(device_ops, result_dir)` 打开。

## 命令行选项

| 选项 | 说明 |
|---|---|
| `--cim-config PATH` | 配置文件，默认 `./cim_config.yaml` |
| `--sdk NAME` | 覆盖 `active_sdk`（也可用环境变量 `CIM_SDK`） |
| `--download` | 收集前下载安装 SDK / 工具链并同步 wheel |
| `--cases DIR` | YAML case 目录 |
| `--device NAME` | 只跑这台设备的测试，其余跳过；多设备并行时每台开一个进程 |
| `--device-password` | 设备 SSH 密码兜底（也可用环境变量 `CIM_DEVICE_PASSWORD`）；优先级低于 `.netrc` |
| `--workdir DIR` | 产物目录 |

## 日志

所有模块通过 `cimalgo.logging.get_logger(__name__)` 输出。环境变量：

- `CIM_LOG_LEVEL`：DEBUG / INFO / WARNING，默认 INFO
- `CIM_LOG_FILE`：设置后同时写文件
- `CIM_LOG_COLOR=0`：关闭颜色

pytest 下日志进入 capture，失败的测试会随报告展示；`pytest -s` 实时查看。

## 目录结构

```
.
├── src/
│   └── cimalgo/
│       ├── config/            # YAML 加载、校验、数据模型 (ProjectConfig / DeviceConfig / ...)
│       ├── env/               # 下载解压、工具链权限、wheel 同步、环境变量 (SetEnv)
│       ├── remote/            # paramiko SSH 连接、设备锁、SDK 部署
│       ├── drivers/           # DeviceOps 契约 (base.py)、各芯片驱动 (bw01d.py)、InferSession
│       ├── pipeline/          # case 定义、假数据、编译/映射工具链、可复用 steps
│       ├── analyzer/          # 精度指标与绘图
│       ├── logging/           # 带缩进和标签的 logger
│       └── pytest_plugin.py   # 命令行选项、会话配置、fixture、case 收集
├── tests/
│   ├── unit/                  # 不依赖设备和 SDK，供 CI 运行
│   └── hardware/              # 需要设备，通过 -m hardware 显式选择
└── cases/                     # YAML case
```

## 扩展

**新芯片**：在 `env/set_env.py` 加 `class XSetEnv(SetEnv, chip="x")`，新建 `drivers/x.py` 写 `class XDeviceOps(DeviceOps, driver="x")` 并在 `drivers/__init__.py` 导入。SDK 的 import 全部放在 `drivers/_sdk.py`（共享的延迟导入和错误包装）和 `drivers/x.py` 内，其他层不感知。

**新指标**：在 `LayerAnalyzer.summary()` 增加一项测试指标，YAML 的 `thresholds` 即可引用。

## 开发

```bash
ruff check . && ruff format .
mypy
pytest tests/unit
```
