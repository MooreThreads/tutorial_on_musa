# GPU Monitor for MTT (Mthreads)

一个轻量级的 GPU 监控工具，用于实时采集和监控 Mthreads GPU 的性能指标。

## 功能特性

- 🚀 **实时监控**：每5秒自动刷新 GPU 信息
- 🔔 **阈值告警**：支持温度和显存占比的告警机制
- 📊 **CSV 日志**：自动记录 GPU 数据到 CSV 文件
- 🔄 **灵活使用**：支持单次读取和循环监控两种模式
- 🧵 **多线程**：后台线程处理，不阻塞主程序
- 📈 **数据导出**：支持 Dict 和对象两种格式获取数据

## 环境要求

- Python 3.6+
- `mthreads-gmi` 命令可用

## 安装

```bash
# 直接使用（无需额外依赖）
python3 mthreads_gpu_monitor.py
```

## 使用方法

### 方式1：单次读取 GPU 信息

```python
from mthreads_gpu_monitor import GPUMonitor

# 创建监控对象
monitor = GPUMonitor()

# 读取一次 GPU 信息
monitor.update()

# 获取所有 GPU 信息（List[Dict] 格式）
all_gpus = monitor.to_dict()
print(all_gpus)

# 获取单张 GPU 信息
gpu0 = monitor.get_gpu(0)
print(f"GPU 0 温度: {gpu0.temperature}°C")

# 获取多张 GPU 信息
gpus = monitor.get_gpu([0, 1, 2])
for gpu in gpus:
    print(gpu)
```

### 方式2：循环监控（后台自动刷新）

```python
from mthreads_gpu_monitor import GPUMonitor

# 创建监控对象（配置告警阈值和CSV日志）
monitor = GPUMonitor(
    refresh_interval=5,           # 刷新间隔（秒）
    csv_path="gpu_metrics.csv",   # CSV 日志文件路径
    alert_config={
        "temperature": 80,              # 温度告警阈值（°C）
        "memory_used_ratio": 0.9,       # 显存占比告警阈值（90%）
    },
)

# 启动后台监控线程
monitor.start()

# 主程序继续执行（监控在后台运行）
import time
time.sleep(60)

# 停止监控
monitor.stop()
```

### 方式3：自定义告警回调

```python
from mthreads_gpu_monitor import GPUMonitor, GPUInfo

def custom_alert(gpu: GPUInfo, msg: str):
    """自定义告警处理函数"""
    print(f"【自定义告警】{msg}")
    # 可以在这里发送邮件、钉钉等

monitor = GPUMonitor(
    refresh_interval=5,
    csv_path="gpu_metrics.csv",
    alert_config={
        "temperature": 80,
        "memory_used_ratio": 0.9,
    },
    alert_callback=custom_alert,  # 传入自定义回调函数
)

monitor.start()
```

## 类和方法说明

### `GPUInfo` 类

GPU 信息的数据类，包含以下属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `index` | int | GPU 索引号 |
| `model` | str | GPU 型号 |
| `temperature` | float | 温度（°C） |
| `power` | float | 功耗（W） |
| `utilization` | float | GPU 利用率（%） |
| `memory_total` | float | 显存总量（MiB） |
| `memory_used` | float | 显存已用（MiB） |
| `memory_used_ratio` | float | 显存占比（0.0-1.0） |

#### 方法

- `to_dict()` - 返回 Dict 格式的数据
- `__repr__()` - 返回对象的字符串表示

### `GPUMonitor` 类

GPU 监控器主类。

#### 初始化参数

```python
GPUMonitor(
    refresh_interval: int = 5,                              # 刷新间隔（秒）
    csv_path: Optional[str] = None,                        # CSV 日志路径
    alert_config: Optional[Dict[str, float]] = None,       # 告警配置
    alert_callback: Optional[Callable[[GPUInfo, str], None]] = None  # 告警回调
)
```

#### 主要方法

| 方法 | 说明 |
|------|------|
| `update()` | 立即更新一次 GPU 信息（含告警和CSV记录） |
| `start()` | 启动后台监控线程（定时调用 update）|
| `stop()` | 停止后台监控线程 |
| `to_dict()` | 返回所有 GPU 信息的 Dict 列表 |
| `get_gpu(index)` | 按索引获取单张或多张 GPU 信息 |

## 示例输出

### 方式1：单次读取
```
所有 GPU 信息:
[
    {
        'index': 0,
        'model': 'MTT S4000',
        'temperature': 75.0,
        'power': 274.7,
        'utilization': 0.0,
        'memory_total': 49152.0,
        'memory_used': 516.0,
        'memory_used_ratio': 0.0105
    },
    ...
]

第0号 GPU 的 memory_total 属性:
49152.0
```

### 方式2：循环监控
```
GPU 监控程序已启动...
每 5 秒刷新一次，温度 ≥80°C 或显存占比 ≥90% 时告警
CSV日志保存到: gpu_metrics.csv
按 Ctrl+C 停止监控

[2026-01-13 12:29:55] GPU Monitor Status:
--------------------------------------------------------------------------------
GPU 0 (MTT S4000):
  温度:   75.0°C  | 功耗:  274.7W
  显存:     516/  49152 MiB (  1.0%)
  利用率:   0.0%
GPU 1 (MTT S4000):
  温度:   63.0°C  | 功耗:  253.9W
  显存:     516/  49152 MiB (  1.0%)
  利用率:   0.0%
...
```

## CSV 日志格式

自动生成的 CSV 文件包含以下列：

```csv
timestamp,gpu_index,model,temperature,utilization,memory_used,memory_total,power
2026-01-13T12:29:55.123456,0,MTT S4000,75.0,0.0,516,49152,274.7
2026-01-13T12:29:55.123456,1,MTT S4000,63.0,0.0,516,49152,253.9
```

## 告警机制

### 默认告警

当以下条件满足时，会触发告警：

1. **温度告警**：`temperature >= alert_config["temperature"]`
2. **显存告警**：`memory_used_ratio >= alert_config["memory_used_ratio"]`

### 告警输出

```
[ALERT] GPU 0 temperature exceeded | temp=85.5C mem_ratio=0.55
[ALERT] GPU 2 memory exceeded | temp=70.0C mem_ratio=0.92
```

### 自定义告警

通过 `alert_callback` 参数传入自定义函数处理告警：

```python
def send_alert_email(gpu: GPUInfo, msg: str):
    # 发送邮件
    pass

monitor = GPUMonitor(alert_callback=send_alert_email)
```

## 常见问题

### Q: 如何在实际程序中集成此监控工具？

A: 启动监控线程后，主程序可以继续执行其他任务，监控在后台运行：

```python
monitor = GPUMonitor(...)
monitor.start()

# 主程序代码
for i in range(100):
    # 处理任务...
    pass

monitor.stop()
```

### Q: 如何获取最新的 GPU 数据？

A: 在循环监控模式下，访问 `monitor.gpus` 即可获取最新数据：

```python
monitor.start()
time.sleep(10)
for gpu in monitor.gpus:
    print(gpu.temperature)
```

### Q: 支持多进程吗？

A: 支持。每个 GPUMonitor 实例独立运行，可创建多个实例进行监控。

### Q: 告警阈值可以动态修改吗？

A: 可以，修改 `monitor.alert_config` 字典即可：

```python
monitor.alert_config["temperature"] = 90  # 修改温度告警阈值
```

## 故障排除

### 错误：`mthreads-gmi: command not found`

确保 `mthreads-gmi` 命令已正确安装并在 PATH 中。

### 数据为空

检查是否有 Mthreads GPU 硬件连接，运行：
```bash
mthreads-gmi -q --json
```

### CSV 文件权限问题

确保对 CSV 文件路径的目录有写权限。

## 许可证

MIT

## 作者

wangkang
