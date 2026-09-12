# COMTRADE 解析模块

> 故障录波智能辅助分析验证平台（一期）· 解析层
> 合同依据：第一条 2.(1)① ②

---

## 一、职责边界

本模块只做一件事：**把 COMTRADE 文件变成下游可以直接用的统一数据结构**。

### 做（合同第一条 2.(1)①② 明确要求）

| 合同/需求条款 | 实现位置 |
|---|---|
| 支持 `.cfg` 配置文件解析 | `comtrade/cfg_parser.py` |
| 支持 `.dat` 数据文件读取 | `comtrade/dat_parser.py` |
| 自动识别录波通道信息 | `comtrade/channels.py` |
| 获取采样频率及时间轴信息 | `comtrade/timebase.py` |
| 建立统一内部数据模型 | `comtrade/models.py` |
| 提供异常文件检测及错误提示机制 | `comtrade/diagnostics.py` |
| 录波数据由原始文件到系统内部分析数据的自动转换 | `comtrade/convert.py` |
| cfg/dat 自动匹配、完整性检查、非法格式检测 | `comtrade/reader.py`（F-01/F-03/F-04/F-05） |

覆盖 COMTRADE **1991 / 1999 / 2013** 三个版本，**ASCII / BINARY / BINARY32 / FLOAT32** 四种编码。

### 不做（属于其它模块，本合同条款未授权解析层承担）

| 不做的事 | 归属 |
|---|---|
| 电气特征计算（有效值、峰值、变化率、序分量） | 算法模块（E） |
| 故障类型与相别判断、规则库 | 算法模块（E） |
| 波形抽稀、降采样、显示 | 客户端模块（F）。**解析层不为了让界面画得动而丢弃数据精度** |
| PDF 报告生成 | 测试交付模块（G） |
| 历史案例库读写（SQLite） | 案例模块（F） |
| 数据修复（插值、补零、滤波） | 一律不做 —— 畸形数据如实记录并报警，不修改原始信息 |

> 唯一两个例外是 `Recording.index_range()` / `slice_time()`：它们只做**索引计算**（按时间区间求采样点下标），不复制、不改变数据，属于"模块调用接口"的一部分，不是显示层的抽稀。

---

## 二、目录结构

```
comtrade_analysis/
├── comtrade/                 解析模块（交付物本体）
│   ├── __init__.py           对外 API 汇总
│   ├── __main__.py           命令行入口（现场排查用）
│   ├── models.py             ★ 统一内部数据模型（解析层的对外契约）
│   ├── diagnostics.py        ★ 错误码、诊断分级、异常类型
│   ├── options.py            解析选项（把现场文件的坑变成可配置项）
│   ├── encoding.py           文本编码兜底（UTF-8 / GBK / GB18030）
│   ├── units.py              单位归一化（kV→V、kA→A）
│   ├── cfg_parser.py         .cfg 解析（版本自适应）
│   ├── dat_parser.py         .dat 解析（四种编码）
│   ├── convert.py            原始值 → 工程值
│   ├── channels.py           通道角色识别
│   ├── timebase.py           时间轴生成
│   └── reader.py             ★ 对外主入口
├── tools/
│   └── make_samples.py       合成样例生成器（兼容性测试样例库）
└── tests/
    ├── run_tests.py          轻量测试运行器（不依赖 pytest）
    ├── _util.py              测试辅助
    ├── test_units_and_channels.py
    ├── test_cfg.py
    ├── test_dat.py
    ├── test_convert.py
    ├── test_timebase.py
    ├── test_reader.py
    └── test_roundtrip.py     对全部样例做端到端回归
```

依赖：**Python ≥ 3.11**、**numpy**。仅此两项（测试与样例生成也只用标准库 + numpy）。

---

## 三、快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 命令行（开发调试 / 现场排查）

```bash
python -m comtrade 故障录波.cfg           # 摘要 + 诊断
python -m comtrade 故障录波.dat           # 拖入任意一个文件都能自动配对
python -m comtrade 故障录波.cfg --json    # 结构化输出，供其它工具消费
python -m comtrade 故障录波.cfg --quiet   # 只输出诊断
```

客户发来一个"打不开"的文件时，先跑这个命令看诊断信息。

### Python API

```python
from comtrade import load_recording, ChannelRole

rec = load_recording(r"D:\cases\fault1.cfg")

rec.meta.station_name          # '110kV 某某变'
rec.meta.version               # ComtradeVersion.V1999
rec.meta.data_type             # DataFileType.BINARY
rec.sample_count               # 1600
rec.duration                   # 0.39975

ia = rec.require_role(ChannelRole.IA)   # 按角色取通道，不依赖通道名
ia.values                       # numpy 数组，已换算成一次值、单位已归一（A）
ia.unit                         # 'A'

rec.time_axis                   # 时间轴（秒），相对第一个采样点
rec.digital_channels[0].transitions()   # 开关量变位位置

for d in rec.diagnostics:
    print(d)
```

**永不抛异常的版本**（批量处理、回归测试、客户样例批量验证用）：

```python
from comtrade import try_load_recording

rec, diagnostics = try_load_recording(path)
if rec is None:
    for d in diagnostics:
        print(d)
```

单文件导入（界面用）则捕获异常，把诊断展示给用户：

```python
from comtrade import load_recording, ComtradeParseError

try:
    rec = load_recording(path)
except ComtradeParseError as exc:
    show_error_dialog("\n".join(str(d) for d in exc.diagnostics))
```

---

## 四、统一内部数据模型（下游模块的接口契约）

```
Recording（录波对象）
├── meta: Metadata
│   ├── station_name / device_id / revision_year / version
│   ├── source_cfg / source_dat / cfg_sha256 / dat_sha256
│   ├── line_frequency
│   ├── sample_rate_segments: [SampleRateSegment(rate_hz, end_sample)]
│   ├── start_time / trigger_time
│   ├── data_type / time_mult / time_base_seconds
│   ├── timezone_code / local_code / time_quality_code / leap_second   (2013)
│   ├── header_text (.hdr) / info_text (.inf)
│   └── total_channels / analog_count / digital_count / sample_count
├── analog_channels: [AnalogChannel]
│   ├── 文件元数据: index, declared_no, name, phase_raw, circuit, unit_raw,
│   │               a, b, skew_us, raw_min, raw_max, primary, secondary, ps
│   ├── 归一化信息: unit, quantity, role, role_confidence, role_source
│   └── 解析结果:   values（工程量）, raw_values(可选), invalid_mask,
│                   scaling_applied, ratio_applied
├── digital_channels: [DigitalChannel]
│   └── index, declared_no, name, phase_raw, circuit, normal_state, values(bool)
├── time_axis: np.ndarray     时间轴（秒），相对第一个采样点，单调递增
├── sample_numbers: np.ndarray
└── diagnostics: [Diagnostic] 全部诊断信息
```

### 四条必须遵守的约定

1. **只读**。解析结果一经产出不再修改。下游做滤波、重采样、抽稀等任何变换，都必须生成新对象。这是 K-01（自动保存）与 NF-14（重复分析结果稳定）的前提。
2. **单位已归一化**。电压一律 V，电流一律 A。原始单位字符串保留在 `unit_raw` 里供报告展示。
3. **数值默认是一次值**。cfg 提供 PS 与变比时，`values` 已按 PS 换算到一次侧；1991 版没有这些字段，会记录 `CHN-004` 提示数值可能仍是二次值。
4. **不承担业务**。这里没有有效值、峰值、序分量、故障类型 —— 那些属于算法模块（E）。

### 通道角色（`ChannelRole`）

下游**必须按角色取通道，不要匹配通道名字符串** —— 同一个 A 相电流在不同录波器里可能叫 `Ia` / `IA` / `I a` / `A相电流` / `IL1` / `Ia1`。

```
UA UB UC UAB UBC UCA U0       电压
IA IB IC I0                   电流
OTHER / UNKNOWN               未识别，需人工映射
```

```python
roles = rec.by_role_map()                 # 角色 → 通道，只含识别成功的
missing = [r.value for r in (ChannelRole.IA, ChannelRole.IB, ChannelRole.IC)
           if r not in roles]             # 算法模块据此判断三相量是否齐全
rec.unresolved_channels()                 # 需要人工映射的通道
```

识别置信度低于 1.0 或标为 `UNKNOWN` 的通道，界面应提供人工映射入口（对应分工方案里 D 的"通道映射方案"）。

---

## 五、诊断与错误码

三级分级（`Severity`）：

| 等级 | 含义 | 处理 |
|---|---|---|
| `FATAL` | 无法产出有效数据 | 抛 `ComtradeParseError`，界面展示诊断 |
| `WARNING` | 解析继续，但结果可能受影响 | 展示给用户，建议人工确认 |
| `INFO` | 提示，不影响结果 | 写日志即可 |

**架构保证 NF-12（错误文件不崩溃）**：可恢复问题一律记录诊断后继续，不抛异常；只有"无法产出任何有效数据"才中断。调用方只需在入口处 catch 一次。

| 前缀 | 范围 | 示例 |
|---|---|---|
| `FIL-` | 文件层面 | `FIL-003` cfg/dat 未配对 |
| `CFG-` | 配置解析 | `CFG-012` 日期二义性、`CFG-017` 比例系数 a 为 0 |
| `DAT-` | 数据解析 | `DAT-002` 文件大小与声明不符、`DAT-009` 疑似已是工程量 |
| `CHN-` | 通道识别 | `CHN-001` 角色无法识别、`CHN-004` 缺变比信息 |
| `TIM-` | 时间轴 | `TIM-001` 时间轴非单调、`TIM-002` 采样率与采样时标不一致 |
| `CHK-` | 结果自检 | `CHK-003` 三相电流不完整 |

完整错误码表见 `comtrade/diagnostics.py` 的 `Code` 类。**错误码一旦发布不应修改含义** —— 测试用例与界面提示都依赖它。

---

## 六、解析选项

```python
from comtrade import ParseOptions, AsciiScaling

opts = ParseOptions(
    ascii_scaling=AsciiScaling.AUTO,   # ASCII 的 a/b 换算策略：auto / always / never
    apply_ratio_conversion=True,       # 是否按 PS 把数值换算到一次值
    mask_missing_values=True,          # 是否把缺失值哨兵标为 NaN
    mask_out_of_range=False,           # 是否按 min/max 剔除越界点（默认关，见下）
    keep_raw_values=False,             # 是否保留原始采样值（内存翻倍）
    value_dtype="float64",             # 换取 float32 可显著降低大文件内存
)
rec = load_recording(path, opts)
```

`mask_out_of_range` 默认关闭是有意的：`min/max` 是文件里的元数据，现场常常写错，按它删数据是破坏性的。默认只报诊断、不删数据。

---

## 七、版本与编码支持矩阵

| 项目 | 1991 | 1999 | 2013 |
|---|---|---|---|
| 首行字段数 | 2 | 3（+版本年份） | 3 |
| 模拟通道字段数 | **10** | 13（+ primary/secondary/PS） | 13 |
| 开关量通道字段数 | **3** | 5（+ ph/ccbm） | 5 |
| `time_mult` | 无 | 有 | 有 |
| dat 类型 | ASCII / BINARY | 同左 | + **BINARY32 / FLOAT32** |
| 附加文件 | `.hdr` | + `.inf` | + `.cff`（**本期未实现**，会明确报 `FIL-006`） |
| 时间码行 | 无 | 无 | `time_code,local_code` + `tmq_code,leapsec` |
| 时间精度 | 微秒 | 微秒 | 可到纳秒 |

解析器对 1991/1999 的字段数差异**全部容忍**（接受 10 或 13、3 或 5）：现场有版本年份写错但字段数正确、或反之的文件，不能因为元数据写错就判定文件非法。版本判定以**首行字段个数**为准，不以版本年份数值为准。

---

## 八、已知的二义性与处理策略

这一节是给算法模块（E）、技术顾问（B）和电力专家看的 —— **有四件事按标准无法唯一确定，实现时做了显式选择，都需要在客户样例到位后确认**。

### 1. 零序电流口径：`I0` 还是 `3I0`（差 3 倍）★ 需确认

严格定义的 `I0 = (Ia+Ib+Ic)/3`，工程上常用的 `3I0 = Ia+Ib+Ic`（中性线/接地线上实际流过的电流），两者相差 3 倍。

- 本模块**只把通道标成 `I0` 角色，不对数值做任何 /3 或 ×3 处理**。
- 需求文档附录一写的 `I0 = Ia+Ib+Ic` 实际上是 `3I0`。
- 规则表里"零序电流 ≥ 0.1 倍"按哪种口径整定，**必须与电力专家确认后在规则配置里写明白**，否则阈值会差 3 倍。

### 2. ASCII 数据的 a/b 换算

标准与主流实现（python-comtrade / comtrade-rs / GSF / pycomtrade）都把 ASCII 的模拟量当原始值，同样施加 `a×raw+b`。但现场确实存在直接写工程量的 ASCII 文件，盲目换算会得到**静默错误**（差几百倍且不报错）。

本模块默认 `AUTO`：用数据跨度与 cfg 声明的量程跨度比对来自动判定，判定结果记入诊断（`DAT-009`），可用 `ascii_scaling=always/never` 覆盖。调研的四个开源解析器**都没有做这个判定**。

### 3. FLOAT32 的数值语义

FLOAT32 的数值通常已经是工程量，本模块按此处理（跳过 a/b 换算，只做单位与 PS 变比换算），并记录 `DAT-011` 提示。

### 4. 日期格式 `日/月` vs `月/日`

标准是 `日/月/年`，但现场有写成 `月/日/年` 的。判定规则：第一个数 > 12 才能确定为 `日/月`；两个数都 ≤ 12 时无法判定，按标准取 `日/月` 并记录 `CFG-012`。建议在导入界面上把解析出的时间显示出来让用户确认一次。

---

## 九、测试与样例

### 运行测试

```bash
python tests/run_tests.py          # 124 个测试，不需要 pytest
python -m pytest tests             # 装了 pytest 也可以
```

### 样例文件

项目目前没有客户真实样例。`tools/make_samples.py` 会合成一套覆盖矩阵完整的录波文件：

```bash
python tools/make_samples.py --out tests/fixtures/generated
```

生成的是**合成的 A 相单相接地故障录波**（110kV 线路、CT 400/1、PT 110kV/100V，故障后 A 相电流突增 5 倍、电压降至 30%），量纲与真实录波一致，算法模块（E）后续可直接用这些文件验证判据。

| 样例 | 覆盖点 |
|---|---|
| `v1999_binary` | 标准 1999 版二进制（13 字段 / 5 字段 / time_mult） |
| `v1991_ascii` | 标准 1991 版 ASCII（10 字段 / 3 字段 / 无 time_mult） |
| `v2013_binary32` | 2013 版 int32 + time_code/tmq_code 行 |
| `v2013_float32` | 2013 版 float32（数值为工程量） |
| `v1999_multirate` | 变频采样（1000Hz→4000Hz），验证 endsamp 累计编号语义 |
| `v1999_missing` | 缺失值哨兵（0x8000） |
| `v1999_chinese_gbk` | 中文通道名 + GBK 编码 |
| `v1999_ascii_prescaled` | ASCII 内容已是工程量，验证不二次换算 |
| `v1999_noisy` | 3% 随机噪声 |

客户样例到位后应加入回归：同一份文件，解析结果必须逐点一致。

---

## 十、实现要点（维护者须知）

以下几条是踩过的坑，改动代码时不要"顺手优化"掉：

1. **`endsamp` 是累计编号，不是段内点数。** `[(1000,400), (4000,1600)]` 表示第 1~400 点按 1000Hz，第 401~1600 点按 4000Hz。时间轴生成的两段必须连续衔接（`TIM-002` 交叉校验专门防这个）。
2. **二进制开关量是按位打包的**，每 16 个通道一个 uint16，最低位对应该组第 1 个通道。不是一个通道一个字节。
3. **字节序固定小端**，解析后用 `min/max` 做范围校验 —— 字节序判断错误时数值会大幅越界，立刻能发现。
4. **缺失值必须在 a/b 换算之前屏蔽**，否则哨兵会被换算成一个"看似正常"的值。
5. **`DAT-002`（文件大小与声明记录数不符）是最有价值的诊断**，实现时务必保留：它能一次性暴露通道数解析错误、版本判定错误、文件截断三类问题。
6. **诊断按文件聚合，不要逐通道刷屏。** 一个根因（如 ASCII 已换算）只报一条，列出受影响的通道名。
7. **`declared_no` 不可信。** 现场通道序号有从 0 起、跳号、重复的情况，索引一律用读取顺序 `index`。
8. **抽样/降采样不要加进解析层。** 波形显示画不动是 F 模块的事；解析层丢数据会连带毁掉特征计算精度。

---

## 十一、待办

- [ ] 客户样例到位后，用真实文件跑 `python -m comtrade` 做兼容性验证（对应需求 T03/NF-31）
- [ ] 2013 版 `.cff` 单文件格式（当前明确报 `FIL-006`；若客户样例中存在需评估是否列入本期）
- [ ] 与电力专家确认零序电流口径（`I0` vs `3I0`）
- [ ] 通道人工映射界面（本模块已把未识别通道隔离到 `UNKNOWN` 并给出 `CHN-001`，映射 UI 属 F 模块）
- [ ] 需求文档中规则库编号 `W-01~W-07` 与波形模块 `W-01~W-07` 重号，建议改为 `R-01~R-07`
