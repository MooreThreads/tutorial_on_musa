#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: wangkang

"""
GPU Monitor for MTT (mthreads-gmi)

Features:
- Periodic GPU info refresh
- CSV logging
- Threshold alerts (temperature / memory)
"""

import json
import subprocess
import threading
import time
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Union


class GPUInfo:
    def __init__(
        self,
        index: int,
        model: str,
        memory_total: float,
        memory_used: float,
        utilization: float,
        temperature: float,
        power: float,
    ):
        self.index = index
        self.model = model
        self.memory_total = memory_total
        self.memory_used = memory_used
        self.utilization = utilization
        self.temperature = temperature
        self.power = power


    @property
    def memory_used_ratio(self) -> float:
        if self.memory_total <= 0:
            return 0.0
        return self.memory_used / self.memory_total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "model": self.model,
            "memory_total": self.memory_total,
            "memory_used": self.memory_used,
            "memory_used_ratio": self.memory_used_ratio,
            "utilization": self.utilization,
            "temperature": self.temperature,
            "power": self.power,
        }

    def __repr__(self) -> str:
        return (
            f"GPUInfo(index={self.index}, model='{self.model}', "
            f"util={self.utilization}%, temp={self.temperature}C, "
            f"memory_used={self.memory_used}MiB, "
            f"memory_total={self.memory_total}MiB, "
            f"power={self.power}W)"
        )


class GPUMonitor:
    def __init__(
        self,
        refresh_interval: int = 5,
        csv_path: Optional[str] = None,
        alert_config: Optional[Dict[str, float]] = None,
        alert_callback: Optional[Callable[[GPUInfo, str], None]] = None,
    ):
        """
        refresh_interval: 刷新间隔（秒）
        csv_path: CSV 保存路径（None 表示不保存）
        alert_config:
            {
                "temperature": 80,
                "memory_used_ratio": 0.9
            }
        """
        self.command = ["mthreads-gmi", "-q", "--json"]
        self.refresh_interval = refresh_interval
        self.csv_path = csv_path
        self.alert_config = alert_config or {}
        self.alert_callback = alert_callback

        self.gpus: List[GPUInfo] = []

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _extract_float(self, value: Any, unit: str = "") -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value).rstrip(unit).strip())

    def _run_command(self) -> Optional[List[GPUInfo]]:
        try:
            result = subprocess.run(
                self.command,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                print("mthreads-gmi failed:", result.stderr)
                return None

            data = json.loads(result.stdout)
            gpus: List[GPUInfo] = []

            for gpu in data.get("GPU", []):
                gpus.append(
                    GPUInfo(
                        index=int(gpu.get("Index", -1)),
                        model=gpu.get("Product Name", "Unknown"),
                        memory_total=self._extract_float(
                            gpu.get("FB Memory Usage", {}).get("Total", 0), "MiB"
                        ),
                        memory_used=self._extract_float(
                            gpu.get("FB Memory Usage", {}).get("Used", 0), "MiB"
                        ),
                        utilization=self._extract_float(
                            gpu.get("Utilization", {}).get("Gpu", 0), "%"
                        ),
                        temperature=self._extract_float(
                            gpu.get("Temperature", {}).get("GPU Current Temp", "0C"), "C"
                        ),
                        power=self._extract_float(
                            gpu.get("Power Readings", {}).get("Power Draw ", "0W"), "W"
                        ),
                    )
                )
            return gpus

        except Exception as e:
            print("GPU query error:", e)
            return None

    def update(self):
        """更新GPU信息并处理告警和CSV日志"""
        gpus = self._run_command()
        if gpus:
            self.gpus = gpus
            self._check_alerts()
            if self.csv_path:
                self._save_csv()

    def to_dict(self) -> List[Dict[str, Any]]:
        """
        返回所有GPU信息（dict格式）
        """
        return [gpu.to_dict() for gpu in self.gpus]
    
    def get_gpu(self, index: Union[int, List[int]]) -> Optional[GPUInfo]:
        """
        按index获取单张GPU
        """
        if isinstance(index, int):
            return self.gpus[index] if 0 <= index < len(self.gpus) else None
        elif isinstance(index, list):
            return [self.gpus[i] for i in index if 0 <= i < len(self.gpus)]
        return None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        while not self._stop_event.is_set():
            self.update()
            time.sleep(self.refresh_interval)

    def _save_csv(self):
        file_exists = False
        try:
            with open(self.csv_path, "r"):
                file_exists = True
        except FileNotFoundError:
            pass

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "gpu_index",
                    "model",
                    "temperature",
                    "utilization",
                    "memory_used",
                    "memory_total",
                    "power",
                ])

            ts = datetime.now().isoformat()
            for gpu in self.gpus:
                writer.writerow([
                    ts,
                    gpu.index,
                    gpu.model,
                    gpu.temperature,
                    gpu.utilization,
                    gpu.memory_used,
                    gpu.memory_total,
                    gpu.power,
                ])


    def _check_alerts(self):
        for gpu in self.gpus:
            if "temperature" in self.alert_config:
                if gpu.temperature >= self.alert_config["temperature"]:
                    self._alert(gpu, "temperature")

            if "memory_used_ratio" in self.alert_config:
                if gpu.memory_used_ratio >= self.alert_config["memory_used_ratio"]:
                    self._alert(gpu, "memory")

    def _alert(self, gpu: GPUInfo, alert_type: str):
        msg = (
            f"[ALERT] GPU {gpu.index} {alert_type} exceeded | "
            f"temp={gpu.temperature}C "
            f"mem_ratio={gpu.memory_used_ratio:.2f}"
        )
        if self.alert_callback:
            self.alert_callback(gpu, msg)
        else:
            print(msg)



if __name__ == "__main__":

    # 方式1: 只读取一次 GPU 信息

    monitor = GPUMonitor()
    monitor.update()  # 直接调用 update() 读取一次

    # 一次性打印所有GPU信息（List[dict]格式）
    print("=== 方式1: 只读取一次 GPU 信息 ===")
    print("所有 GPU 信息:")
    print(monitor.to_dict(), "\n")

    # 打印第0号GPU信息（dict格式）
    print("第0号 GPU 信息:")
    print(monitor.gpus[0].to_dict(), "\n")

    # 使用 get_gpu 方法获取 GPUInfo 对象, 并打印其属性
    print("第0号 GPU 的 memory_total 属性:")
    print(monitor.get_gpu(0).memory_total, "\n")

    # 获取多张GPU信息
    print(monitor.get_gpu([0, 1]), "\n")

    
    # # 方式2: 循环监控（每5秒刷新一次）
    print("\n=== 方式2: 循环监控 ===")
    monitor = GPUMonitor(
        refresh_interval=5,
        csv_path="gpu_metrics.csv",
        alert_config={
            "temperature": 80,
            "memory_used_ratio": 0.9,
        },
    )
    print("GPU 监控程序已启动...")
    print("每 5 秒刷新一次，温度 ≥80°C 或显存占比 ≥90% 时告警")
    print("CSV日志保存到: gpu_metrics.csv")
    print("按 Ctrl+C 停止监控\n")

    monitor.start()

    time.sleep(30)  # 你要运行的程序！！！

    monitor.stop()





