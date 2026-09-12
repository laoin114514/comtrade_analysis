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
    ├── test_diagnostics.py   错误码登记完整性（守卫：码不得写死为字面量）
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
   PS 字段**存在但为空或无法识别**时同样不做换算（不猜测数值基准），但按**警告**记录 `CHN-006`——此时变比信息实际在文件里，漏掉换算会让数值偏小变比倍数（CT 400/1 就是 400 倍），而二次值本身看起来完全正常，必须人工确认。
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
| `FIL-` | 文件层面 | `FIL-003` cfg/dat 未配对、`FIL-008` 伴随文件读取失败 |
| `CFG-` | 配置解析 | `CFG-012` 日期二义性、`CFG-017` 比例系数 a 为 0、`CFG-021` 数值字段无法解析 |
| `DAT-` | 数据解析 | `DAT-002` 文件大小与声明不符、`DAT-009` 疑似已是工程量、`DAT-013` 采样号/时标为空（该点无法定位） |
| `CHN-` | 通道识别 | `CHN-001` 角色无法识别、`CHN-004` 缺变比信息、`CHN-005` 一个角色被多个通道占用、`CHN-006` PS 字段读不出来（数值可能偏小变比倍数） |
| `TIM-` | 时间轴 | `TIM-001` 时间轴非单调、`TIM-002` 采样率与采样时标不一致 |
| `CHK-` | 结果自检 | `CHK-003` 三相电流不完整 |
| `SYS-` | 兜底异常（不应出现，出现即缺陷） | `SYS-001` 未预期的错误 |

完整错误码表见 `comtrade/diagnostics.py` 的 `Code` 类。**错误码一旦发布不应修改含义** —— 测试用例与界面提示都依赖它。

界面层按码组织提示语时，请以 `Code` 类为唯一来源：**业务代码中不得把错误码写成字符串字面量**，一律引用 `Code.XXX` 常量。`tests/test_diagnostics.py` 有守卫测试，任何未登记的码字面量都会让测试失败。正常路径也会给出 INFO 级标记（`CFG-OK` / `DAT-OK` / `TIM-OK` / `CFG-ENC`），界面可用它们表示"已识别 / 已解析"。

---

## 六、解析选项

```python
from comtrade import ParseOptions, AsciiScaling

opts = ParseOptions(
    ascii_scaling=AsciiScaling.ALWAYS, # ASCII 的 a/b 换算策略：always（默认）/ never
    time_axis_source="rates",          # 时间轴优先依据：rates（默认）/ timestamps
    apply_ratio_conversion=True,       # 是否按 PS 把数值换算到一次值
    mask_missing_values=True,          # 是否把缺失值哨兵标为 NaN
    mask_out_of_range=False,           # 是否按 min/max 剔除越界点（默认关，见下）
    keep_raw_values=False,             # 是否保留原始采样值（内存翻倍）
    value_dtype="float64",             # 换取 float32 可显著降低大文件内存
)
rec = load_recording(path, opts)
```

`mask_out_of_range` 默认关闭是有意的：`min/max` 是文件里的元数据，现场常常写错，按它删数据是破坏性的。默认只报诊断、不删数据。

`time_axis_source` 是项目内两套解析实现的分歧点（另一套是采样时标优先）。单频文件里两种依据会差 0.04%~0.1% —— 采样率是装置声明的「契约值」，采样时标反映「实际时钟漂移」。**用客户真实文件确定后应固定下来**，否则同一份录波在两处会算出不同的时长。

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

标准与全部主流实现都对 ASCII 的模拟量施加 `a×raw+b` 换算，本模块默认遵循。
现场确实也存在直接写工程量的 ASCII 文件，但**两者无法可靠区分** ——
真实文件里「min/max 声明满量程、实际原始计数很小」是正常现象。

本模块的处理：默认按标准换算；当数据跨度远小于声明量程时给出 `DAT-009` **提示**
（不改变换算行为），确认某文件确实存的是工程量时，用 `ascii_scaling="never"` 显式跳过。

> 早期版本曾用该判据**自动**跳过换算，在 11 组真实样例上误判 10 个通道且全部错向"跳过"，
> 因此改为现在的"默认换算 + 提示"。

### 3. FLOAT32 的数值语义

FLOAT32 通常直接存工程量，cfg 里的 a/b 按惯例写 `a=1, b=0`。
本模块对其一视同仁地走 `a/b → 单位 → PS 变比` 链路；当 a/b 不是单位映射时给出 `DAT-011` 提示。
实测公开样例库中**没有 FLOAT32 样例**，此行为需用客户样例确认。

### 4. 日期格式 `日/月` vs `月/日`

标准是 `日/月/年`，但现场有写成 `月/日/年` 的。判定规则：第一个数 > 12 才能确定为 `日/月`；两个数都 ≤ 12 时无法判定，按标准取 `日/月` 并记录 `CFG-012`。建议在导入界面上把解析出的时间显示出来让用户确认一次。

---

## 九、测试与样例

### 运行测试

```bash
python tests/run_tests.py          # 全量回归（当前 176 个用例），不需要 pytest
python -m pytest tests             # 装了 pytest 也可以
```

### 合成样例（自建）

项目自身没有客户样例。`tools/make_samples.py` 会合成一套覆盖矩阵完整的录波文件：

```bash
python tools/make_samples.py --out tests/fixtures/generated
```

生成的是**合成的 A 相单相接地故障录波**（110kV 线路、CT 400/1、PT 110kV/100V，故障后 A 相电流突增 5 倍、电压降至 30%），量纲与真实录波一致，算法模块（E）后续可直接用这些文件验证判据。

| 样例 | 覆盖点 |
|---|---|
| `v1999_binary` | 标准 1999 版二进制（13 字段 / 5 字段 / time_mult） |
| `v1991_ascii` | 标准 1991 版 ASCII（10 字段 / 3 字段 / 无 time_mult） |
| `v2013_binary32` | 2013 版 int32 + time_code/tmq_code 行 |
| `v2013_float32` | 2013 版 float32（数值为工程量、cfg 按惯例写 a=1,b=0） |
| `v1999_multirate` | 变频采样（1000Hz→4000Hz），验证 endsamp 累计编号语义 |
| `v1999_missing` | 缺失值哨兵（0x8000） |
| `v1999_chinese_gbk` | 中文通道名 + GBK 编码 |
| `v1999_ascii_smallspan` | 原始计数只用到 ±100（cfg 却声明 ±32767）—— 复现 SEL 等装置现场情形 |
| `v1999_noisy` | 3% 随机噪声 |

### 真实公开样例库（强烈建议配置）

合成样例只能证明"实现与对标准的理解一致"，证明不了"与现场文件一致"。
因此支持用真实公开样例库做回归，路径由环境变量指定，**未设置时自动跳过**：

```bash
# 样本库参考：fault-wave-analyzer/fastapi/tests/fixtures/sample_library/
# （12 组来自 GitHub MIT 仓库的样例）
export COMTRADE_SAMPLE_LIBRARY=/path/to/sample_library
python tests/run_tests.py
```

也可以用独立工具做完整核查：

```bash
python tools/validate_library.py --library <样本库目录>
python tools/validate_library.py --library <目录> --cross-check <另一解析模块的父目录>
```

`--cross-check` 会把另一个解析器的输出与本模块逐通道对比，
确认数值差异只来自「单位归一化 × PS 变比」这两步（本模块相对标准实现多做的工作）。

### 实测结果（12 组真实样例 / 186 个通道）

| 指标 | 结果 |
|---|---|
| 样例解析成功 | 11/11（第 12 组只有 cfg 无 dat，按设计报 `FIL-003`） |
| 版本判定 | 1991 / 1999 / 2013 全部正确，含非标年份 1997、2000 |
| 通道数、记录数 | 全部与基线一致 |
| 与对照解析器的数值差异 | 186 个通道中：符合「单位×PS变比」**172 个**，双方均为零 14 个，**不符 0 个** |
| 通道角色识别率 | 121/186 = **65.1%**（其余为直流/频率/计数器/俄文自定义名，按设计标 UNKNOWN 交人工映射） |
| 时间轴 | 7 组与对照完全一致；4 组差 0.04%~0.1%（见 §10 第 9 条，已提供开关） |

真实数据暴露并已修复的缺陷，记录在此以免重蹈：

1. **`nrates=0` 但随后仍给出采样率行**（SEL 的 `0,20700`、Wisp 的 `0,  1360`）。
   不消费这一行会导致其后所有行整体错位一行，表现为"起始时间无法解析 → 数据格式无法识别"的致命错误。
2. **ASCII 的 a/b 换算"自动判定"是错误设计**。原以为可以用"数据跨度 vs 声明量程跨度"
   判断数据是否已是工程量，实测在真实样例上误判 10 个通道、**全部错向"跳过换算"**。
   已改为默认按标准换算，跨度判定降级为纯提示（`DAT-009`）。
3. **编码兜底链被 GB18030 卡死**。GB18030 几乎能解码任意字节序列，
   放在 latin-1 之前会把俄文、葡文文件都"成功"解成乱码汉字：
   俄文 `Неизвестный регистратор` → `Íåèçâåñòíûé ðåãèñòðàòîð`，
   葡文 `Estação de Medição` → `Esta玢o de Medi玢o`。已改为
   CP1251（俄文）→ GB18030（中文）→ Latin-1 的判定顺序，并加了防误判判据。
4. **`endsamp` 是采样号不是条数**。有的装置采样号从 0 起（wisp_example2：0~19679、endsamp=19679），
   直接与记录条数比较会误报。已改为按 `endsamp - 首采样号 + 1` 比较。
5. **通道名后缀过窄**。`IAX`/`IBY`/`IAT`（不同绕组的三相电流）、`VA(kV)`
   （单位写在名称里）原先识别不出，放宽后缀规则后识别率 58.1% → 65.1%。

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
8. **`endsamp` 是采样号不是条数，且各装置写法不统一。** 有 0 起编号的（wisp_example2：0~19679、endsamp=19679）、有把 endsamp 写成总条数的（wisp_example5：0~201、endsamp=202）、也有就是差一条的（wisp_example4）。比较时按 `endsamp - 首采样号 + 1` 并容差 ±1 条，只对量级性偏差报警。
9. **编码判定顺序不能随意调整。** GB18030 几乎能解码任意字节序列，必须放在 CP1251 之后，并加「拉丁文本被误读成汉字」的判据（汉字被夹在两个 ASCII 字母之间），否则俄文、葡文文件会被静默解成乱码。
10. **抽样/降采样不要加进解析层。** 波形显示画不动是 F 模块的事；解析层丢数据会连带毁掉特征计算精度。
11. **PS 字段的"没有"与"为空"必须分开。** 两者都让 `ps is None`，但对下游意义相反：
    没有这个字段（1991 版 / 10 字段行）是无信息可用，按 INFO（`CHN-004`）如实告知即可；
    字段存在却读不出来，说明**变比信息在文件里、只是数值基准不明** —— 漏掉换算会让数值
    偏小变比倍数（CT 400/1 就差 400 倍），而二次值本身看起来完全正常，必须按 WARNING
    （`CHN-006`）告警，并把 `ps_raw` 原文与变比一并写进提示，让人能一眼核对
    "数值是不是恰好小了这么多倍"。早期实现把两者混为一谈，于是这种情况静默输出偏小的数值、
    诊断还宣称文件是 1991 版（与文件头声明矛盾）。参见 `AnalogChannel.ps_raw`。
12. **ASCII 列数不符要区分"部分行"与"全部行"。** 部分行不符 → 跳过坏行继续解析，
    记 `DAT-007` 警告（容错的本意）；**全部行**都不符 → 根因是 cfg 声明与 dat 实际列数不一致，
    致命诊断必须直接说明这一点（实际列数、声明列数、可能的通道数问题），
    不能笼统地报 `DAT-001`"没有可解析的采样行" —— 那会把人引去查 dat 是否损坏，
    而数据行本身可能完好无损。
13. **采样号与采样时标这两列绝不能出现非有限值。** 它们必须能转成整数，而
    `astype(np.int64)` 对 NaN 是**静默转换**（NaN → INT64_MIN = -9223372036854775808，
    只留一个 numpy RuntimeWarning，应用层通常关掉了）：采样号就此失去意义（下游按
    采样号索引会算出无意义结果），落在**首行**还会让记录数校验推算 9.2e18 的期望条数、
    误报"数据可能被截断"，把一份完好文件判成不可信。
    因此 ASCII 空字段"→ NaN"只适用于模拟量/开关量列（第 2 列起）：第 0、1 列为空
    （或写成 `nan`/`inf` 字面量）的行必须在容错解析里显式跳过并记 `DAT-013`；
    `_assemble_ascii` 开头另有一道守卫兜底，保证将来新增路径也不会把 NaN 送进这两列。
14. **单位表的大小写与符号参与语义，不能靠 `lower()` 抹平。** SI 词头 `M`(10⁶) 与
    `m`(10⁻³) 只差大小写：统一小写会让 `MV`（兆伏）撞上 `mV`（毫伏），静默差 10⁹ 倍；
    `µ`/`μ` 若被清洗正则当排版字符删掉，`µV` 会退化成 `V`，差 10⁶ 倍。这两类错误都
    `recognized=True`，**不产生任何诊断**，而取值范围校验用的是缩放前的原始值、拦不住。
    规范写法一律先查 `_CASE_SIGNIFICANT_UNITS`（保留大小写精确匹配），再回落
    `_UNIT_MAP`（小写表）。新增单位时先问一句：这个写法的大小写/符号是否参与语义？
15. **解析期产生的 NaN 与缺失值哨兵同属"缺失的采样点"，都必须进 `invalid_mask`。**
    NaN 既不匹配哨兵、也不会被范围校验捕获（与任何数比较都是 False），漏并进掩码会让
    `has_invalid` / `invalid_count` 谎报"没有无效点"，而下游正是据此判断"该通道能否参与
    计算"（`is_usable` 只看数组非空）—— 结果就是算出 NaN 而全程无提示。哨兵那套占比守卫
    不适用于 NaN：它本来就不是有效数据，不存在"误伤正常值"的问题。

---

## 十一、待办

- [ ] 客户样例到位后，用真实文件跑 `python -m comtrade` 做兼容性验证（对应需求 T03/NF-31）；并据实际文件确定 `time_axis_source` 与 `ascii_scaling` 取值
- [ ] 用客户文件确认 FLOAT32 的数值语义（公开样例库中没有 FLOAT32 样例）
- [ ] 确认 `IG`/`IP`（SEL 等装置）、`U21`/`U32`（欧式线电压命名）、`I1`/`I2`/`I3`（与序分量命名冲突）等通道是否需加入别名表 —— 当前一律标 UNKNOWN 交人工映射
- [ ] 2013 版 `.cff` 单文件格式（当前明确报 `FIL-006`；若客户样例中存在需评估是否列入本期）
- [ ] 与电力专家确认零序电流口径（`I0` vs `3I0`）
- [ ] 通道人工映射界面（本模块已把未识别通道隔离到 `UNKNOWN` 并给出 `CHN-001`，映射 UI 属 F 模块）
- [ ] 需求文档中规则库编号 `W-01~W-07` 与波形模块 `W-01~W-07` 重号，建议改为 `R-01~R-07`
