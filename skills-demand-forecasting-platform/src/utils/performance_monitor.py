"""Advanced performance monitoring and optimization utilities."""

import time
import psutil
import logging
import threading
from contextlib import contextmanager
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation_name,
            "duration_seconds": round(self.duration, 3),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "cpu_usage_percent": round(self.cpu_usage_percent, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat()
        }


class PerformanceMonitor:
    """
    Advanced performance monitoring with memory and CPU tracking.
    Provides insights for optimization and resource management.
    """

    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.active_operations: Dict[str, float] = {}
        self.memory_baseline = self._get_memory_usage()
        self.lock = threading.Lock()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0

    @contextmanager
    def track_operation(self, operation_name: str):
        """Context manager for tracking operation performance."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        peak_memory = start_memory

        # Monitor memory usage in background
        monitoring = True

        def monitor_memory():
            nonlocal peak_memory, monitoring
            while monitoring:
                current_memory = self._get_memory_usage()
                peak_memory = max(peak_memory, current_memory)
                time.sleep(0.1)

        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()

        try:
            with self.lock:
                self.active_operations[operation_name] = start_time

            yield

        finally:
            monitoring = False
            end_time = time.time()
            end_memory = self._get_memory_usage()
            cpu_usage = self._get_cpu_usage()

            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_usage_mb=end_memory - start_memory,
                cpu_usage_percent=cpu_usage,
                peak_memory_mb=peak_memory
            )

            with self.lock:
                self.metrics.append(metrics)
                self.active_operations.pop(operation_name, None)

            # Log performance if duration > 1 second
            if metrics.duration > 1.0:
                logging.info(
                    f"⏱️ {operation_name}: {metrics.duration:.2f}s, "
                    f"Memory: {metrics.memory_usage_mb:+.1f}MB, "
                    f"Peak: {metrics.peak_memory_mb:.1f}MB"
                )

    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        operation_metrics = [m for m in self.metrics if m.operation_name == operation_name]

        if not operation_metrics:
            return {"error": f"No metrics found for operation: {operation_name}"}

        durations = [m.duration for m in operation_metrics]
        memory_usage = [m.memory_usage_mb for m in operation_metrics]

        return {
            "operation": operation_name,
            "execution_count": len(operation_metrics),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_memory_usage": sum(memory_usage) / len(memory_usage),
            "total_memory_impact": sum(memory_usage),
            "last_execution": operation_metrics[-1].to_dict()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics:
            return {"message": "No performance metrics available"}

        total_duration = sum(m.duration for m in self.metrics)
        total_memory = sum(abs(m.memory_usage_mb) for m in self.metrics)

        # Group by operation
        operation_groups = {}
        for metric in self.metrics:
            op = metric.operation_name
            if op not in operation_groups:
                operation_groups[op] = []
            operation_groups[op].append(metric)

        operation_stats = {}
        for op, metrics_list in operation_groups.items():
            durations = [m.duration for m in metrics_list]
            operation_stats[op] = {
                "count": len(metrics_list),
                "total_duration": sum(durations),
                "avg_duration": sum(durations) / len(durations),
                "percentage_of_total": (sum(durations) / total_duration * 100) if total_duration > 0 else 0
            }

        return {
            "total_operations": len(self.metrics),
            "total_duration": round(total_duration, 2),
            "total_memory_usage": round(total_memory, 2),
            "avg_operation_duration": round(total_duration / len(self.metrics), 2),
            "operation_breakdown": operation_stats,
            "performance_recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        if not self.metrics:
            return recommendations

        # Find slowest operations
        slowest_ops = sorted(
            [(m.operation_name, m.duration) for m in self.metrics],
            key=lambda x: x[1], reverse=True
        )[:3]

        for op, duration in slowest_ops:
            if duration > 10:
                recommendations.append(
                    f"Consider optimizing '{op}' operation (taking {duration:.1f}s)"
                )

        # Check memory usage
        high_memory_ops = [m for m in self.metrics if m.peak_memory_mb > 500]
        if high_memory_ops:
            recommendations.append(
                f"High memory usage detected in {len(high_memory_ops)} operations. "
                "Consider batch processing or data streaming."
            )

        # Check for repetitive operations
        operation_counts = {}
        for metric in self.metrics:
            operation_counts[metric.operation_name] = operation_counts.get(metric.operation_name, 0) + 1

        frequent_ops = [(op, count) for op, count in operation_counts.items() if count > 5]
        if frequent_ops:
            recommendations.append(
                "Consider caching results for frequently executed operations: " +
                ", ".join([f"{op} ({count}x)" for op, count in frequent_ops])
            )

        return recommendations

    def export_metrics(self, format: str = "json") -> Any:
        """Export metrics in specified format."""
        if format == "json":
            return {
                "export_timestamp": datetime.now().isoformat(),
                "metrics": [m.to_dict() for m in self.metrics],
                "summary": self.get_summary()
            }
        elif format == "dataframe":
            try:
                import pandas as pd
                return pd.DataFrame([m.to_dict() for m in self.metrics])
            except ImportError:
                return "DataFrame export requires pandas"
        else:
            raise ValueError(f"Unsupported export format: {format}")