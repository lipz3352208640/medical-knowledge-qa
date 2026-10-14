"""Small Prometheus-compatible registry with no mandatory runtime dependency."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
import re


_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")


def _labels(labels: dict[str, object]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in labels.items()))


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[tuple[str, tuple], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple], float] = {}
        self._histograms: dict[tuple[str, tuple], list[float]] = defaultdict(list)

    def inc(self, name: str, value: float = 1, **labels: object) -> None:
        self._check(name)
        with self._lock:
            self._counters[(name, _labels(labels))] += value

    def set(self, name: str, value: float, **labels: object) -> None:
        self._check(name)
        with self._lock:
            self._gauges[(name, _labels(labels))] = value

    def observe(self, name: str, value: float, **labels: object) -> None:
        self._check(name)
        with self._lock:
            self._histograms[(name, _labels(labels))].append(float(value))

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = {key: list(value) for key, value in self._histograms.items()}
        for (name, labels), value in sorted(counters.items()):
            lines.append(f"{name}{self._format_labels(labels)} {value:g}")
        for (name, labels), value in sorted(gauges.items()):
            lines.append(f"{name}{self._format_labels(labels)} {value:g}")
        for (name, labels), values in sorted(histograms.items()):
            for bucket in (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10):
                bucket_labels = dict(labels)
                bucket_labels["le"] = "+Inf" if bucket == 10 else bucket
                count = len(values) if bucket == 10 else sum(item <= bucket for item in values)
                lines.append(f"{name}_bucket{self._format_labels(tuple(sorted(bucket_labels.items())))} {count}")
            lines.append(f"{name}_count{self._format_labels(labels)} {len(values)}")
            lines.append(f"{name}_sum{self._format_labels(labels)} {sum(values):g}")
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
        if not labels:
            return ""
        return "{" + ",".join(f'{key}="{value.replace(chr(34), chr(92)+chr(34))}"' for key, value in labels) + "}"

    @staticmethod
    def _check(name: str) -> None:
        if not _NAME.fullmatch(name):
            raise ValueError(f"Invalid metric name: {name}")


METRICS = MetricsRegistry()
