"""
Log Parsing and Anomaly Injection Utilities.

This module provides log parsing for various log formats and anomaly injection
capabilities for creating synthetic training data.
"""

import re
import random
import hashlib
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Support both package and direct execution modes
if __package__:
    from .models import AnomalyType, DifficultyLevel, LogLine
else:
    # Direct execution mode
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import AnomalyType, DifficultyLevel, LogLine


@dataclass
class ParsedLog:
    """Container for parsed log data."""

    lines: List[LogLine]
    start_time: datetime
    end_time: datetime
    components: List[str]
    severities: List[str]


class LogParser:
    """Parser for various log formats."""

    # Common log format patterns
    PATTERNS = {
        "hdfs": r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z?)\s+(?P<severity>BLOCK\w*|INFO|WARN|ERROR|FATAL)\s+(?P<component>\w+)\s+(?P<message>.*)",
        "apache": r'(?P<ip>[\d.]+)\s+-\s+-\s+\[(?P<timestamp>[^\]]+)\]\s+"(?P<request>[^"]+)"\s+(?P<status>\d{3})\s+(?P<size>\d+|-)',
        "syslog": r"(?P<timestamp>\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(?P<host>\S+)\s+(?P<component>\S+)\[(?P<pid>\d+)\]:\s+(?P<message>.*)",
        "json": r"\{.*\}",
    }

    def __init__(self, log_format: str = "hdfs"):
        """
        Initialize the parser.

        Args:
            log_format: Format of logs ("hdfs", "apache", "syslog", "json")
        """
        self.log_format = log_format

    def parse_line(self, line: str, line_number: int = 0) -> Optional[LogLine]:
        """
        Parse a single log line.

        Args:
            line: Raw log line
            line_number: Line number for ordering

        Returns:
            Parsed LogLine or None if parsing fails
        """
        line = line.strip()
        if not line:
            return None

        try:
            if self.log_format == "hdfs":
                return self._parse_hdfs(line)
            elif self.log_format == "apache":
                return self._parse_apache(line)
            elif self.log_format == "syslog":
                return self._parse_syslog(line)
            elif self.log_format == "json":
                return self._parse_json(line)
            else:
                return self._parse_generic(line)
        except Exception:
            return self._parse_generic(line)

    def _parse_hdfs(self, line: str) -> LogLine:
        """Parse HDFS-style log format."""
        pattern = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z?)\s+(BLOCK\w*|INFO|WARN|ERROR|FATAL)\s+(\w+)\s+(.*)"
        match = re.match(pattern, line)
        if match:
            return LogLine(
                timestamp=match.group(1),
                severity=match.group(2),
                component=match.group(3),
                message=match.group(4),
                raw_line=line,
            )
        return self._parse_generic(line)

    def _parse_apache(self, line: str) -> LogLine:
        """Parse Apache-style log format."""
        pattern = r'([\d.]+)\s+-\s+-\s+\[([^\]]+)\]\s+"([^"]+)"\s+(\d{3})\s+(\d+|-)'
        match = re.match(pattern, line)
        if match:
            timestamp = datetime.strptime(match.group(2), "%d/%b/%Y:%H:%M:%S %z")
            return LogLine(
                timestamp=timestamp.isoformat(),
                severity="INFO" if int(match.group(4)) < 400 else "ERROR",
                component=f"apache_{match.group(4)}",
                message=match.group(3),
                raw_line=line,
            )
        return self._parse_generic(line)

    def _parse_syslog(self, line: str) -> LogLine:
        """Parse syslog-style format."""
        pattern = r"(\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+)\[(\d+)\]:\s+(.*)"
        match = re.match(pattern, line)
        if match:
            return LogLine(
                timestamp=match.group(1),
                severity="INFO",
                component=match.group(3),
                message=match.group(5),
                raw_line=line,
            )
        return self._parse_generic(line)

    def _parse_json(self, line: str) -> LogLine:
        """Parse JSON log format."""
        import json

        try:
            data = json.loads(line)
            return LogLine(
                timestamp=data.get("timestamp", datetime.now().isoformat()),
                severity=data.get("level", "INFO").upper(),
                component=data.get("component", data.get("logger", "unknown")),
                message=data.get("message", ""),
                raw_line=line,
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return self._parse_generic(line)

    def _parse_generic(self, line: str) -> LogLine:
        """Parse generic log format as fallback."""
        # Try to extract common elements
        severity = "INFO"
        if "ERROR" in line:
            severity = "ERROR"
        elif "WARN" in line:
            severity = "WARN"
        elif "FATAL" in line:
            severity = "FATAL"
        elif "DEBUG" in line:
            severity = "DEBUG"

        # Try to extract timestamp
        timestamp_match = re.search(r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}", line)
        timestamp = timestamp_match.group(0) if timestamp_match else datetime.now().isoformat()

        # Try to extract component
        component_match = re.search(r"\[([\w\.]+)\]", line)
        component = component_match.group(1) if component_match else "unknown"

        return LogLine(
            timestamp=timestamp,
            severity=severity,
            component=component,
            message=line,
            raw_line=line,
        )

    def parse_file(self, filepath: str) -> ParsedLog:
        """
        Parse an entire log file.

        Args:
            filepath: Path to the log file

        Returns:
            ParsedLog containing all parsed lines
        """
        lines = []
        timestamps = []

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                parsed = self.parse_line(line, i)
                if parsed:
                    lines.append(parsed)
                    try:
                        timestamps.append(
                            datetime.fromisoformat(parsed.timestamp.replace("Z", "+00:00"))
                        )
                    except (ValueError, TypeError):
                        pass

        return ParsedLog(
            lines=lines,
            start_time=min(timestamps) if timestamps else datetime.now(),
            end_time=max(timestamps) if timestamps else datetime.now(),
            components=list(set(l.component for l in lines)),
            severities=list(set(l.severity for l in lines)),
        )


class AnomalyInjector:
    """
    Injects synthetic anomalies into clean log data.

    Supports multiple anomaly types with configurable intensity
    and randomized placement.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the injector.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)

    def inject(
        self,
        logs: List[LogLine],
        anomaly_type: AnomalyType,
        intensity: float = 0.5,
        seed: Optional[int] = None,
    ) -> Tuple[List[LogLine], Dict[str, Any]]:
        """
        Inject an anomaly into log data.

        Args:
            logs: Original log lines
            anomaly_type: Type of anomaly to inject
            intensity: Anomaly intensity (0.0 to 1.0)
            seed: Optional seed for this injection

        Returns:
            Tuple of (modified logs, ground truth metadata)
        """
        if seed is not None:
            self.rng = random.Random(seed)

        if anomaly_type == AnomalyType.ERROR_SPIKE:
            return self._inject_error_spike(logs, intensity)
        elif anomaly_type == AnomalyType.MEMORY_LEAK:
            return self._inject_memory_leak(logs, intensity)
        elif anomaly_type == AnomalyType.SERVICE_DROPOUT:
            return self._inject_service_dropout(logs, intensity)
        elif anomaly_type == AnomalyType.CASCADE_FAILURE:
            return self._inject_cascade_failure(logs, intensity)
        elif anomaly_type == AnomalyType.LATENCY_DEGRADATION:
            return self._inject_latency_degradation(logs, intensity)
        elif anomaly_type == AnomalyType.AUTH_ANOMALY:
            return self._inject_auth_anomaly(logs, intensity)
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    def _inject_error_spike(
        self, logs: List[LogLine], intensity: float
    ) -> Tuple[List[LogLine], Dict[str, Any]]:
        """Inject error spike anomaly."""
        if not logs:
            return logs, {}

        # Find a suitable component and time window
        components = list(set(l.component for l in logs if l.component != "unknown"))
        if not components:
            components = ["component_a"]

        target_component = self.rng.choice(components)

        # Define time window (middle 60% of logs)
        start_idx = len(logs) // 5
        end_idx = len(logs) * 4 // 5
        window_size = int((end_idx - start_idx) * intensity * 0.5) + 1

        window_start = self.rng.randint(start_idx, end_idx - window_size)
        window_end = window_start + window_size

        # Generate error messages
        modified_logs = []
        for i, log in enumerate(logs):
            if window_start <= i < window_end and log.component == target_component:
                # Inject error lines
                error_rate = intensity * 0.8 + 0.2
                if self.rng.random() < error_rate:
                    error_msg = self._generate_error_message(target_component, log.timestamp)
                    modified_logs.append(error_msg)
            modified_logs.append(log)

        # Adjust indices to account for inserted lines
        actual_window_start = window_start
        actual_window_end = window_start + sum(
            1 for i in range(window_start, window_end) if logs[i].component == target_component
        )

        return modified_logs, {
            "anomaly_type": AnomalyType.ERROR_SPIKE.value,
            "component": target_component,
            "start_time": logs[window_start].timestamp
            if window_start < len(logs)
            else logs[0].timestamp,
            "end_time": logs[min(window_end, len(logs) - 1)].timestamp,
            "intensity": intensity,
        }

    def _inject_memory_leak(
        self, logs: List[LogLine], intensity: float
    ) -> Tuple[List[LogLine], Dict[str, Any]]:
        """Inject memory leak anomaly."""
        if not logs:
            return logs, {}

        components = list(set(l.component for l in logs if l.component != "unknown"))
        if not components:
            components = ["jvm"]

        target_component = self.rng.choice(
            [
                c
                for c in components
                if any(m.lower() in c.lower() for m in ["jvm", "gc", "memory", "heap"])
            ]
            or components
        )

        # Define gradual increase window
        start_idx = len(logs) // 5
        end_idx = len(logs) * 4 // 5
        window_size = end_idx - start_idx

        # Generate memory leak pattern
        modified_logs = []
        leak_values = []

        for i, log in enumerate(logs):
            if start_idx <= i < end_idx:
                # Generate increasing memory values
                progress = (i - start_idx) / window_size
                leak_value = int(500 + progress * 2000 * intensity + self.rng.gauss(0, 50))
                leak_values.append(leak_value)

                # Add memory log line
                memory_msg = self._generate_memory_log(target_component, log.timestamp, leak_value)
                modified_logs.append(memory_msg)
            modified_logs.append(log)

        return modified_logs, {
            "anomaly_type": AnomalyType.MEMORY_LEAK.value,
            "component": target_component,
            "start_time": logs[start_idx].timestamp if start_idx < len(logs) else logs[0].timestamp,
            "end_time": logs[min(end_idx, len(logs) - 1)].timestamp,
            "intensity": intensity,
            "peak_memory": max(leak_values) if leak_values else 0,
        }

    def _inject_service_dropout(
        self, logs: List[LogLine], intensity: float
    ) -> Tuple[List[LogLine], Dict[str, Any]]:
        """Inject service dropout anomaly (absence of expected logs)."""
        if not logs:
            return logs, {}

        components = list(set(l.component for l in logs if l.component != "unknown"))
        if not components:
            return logs, {}

        target_component = self.rng.choice(components)

        # Find contiguous segment of this component's logs
        component_indices = [i for i, l in enumerate(logs) if l.component == target_component]
        if len(component_indices) < 10:
            return logs, {}

        # Choose dropout window
        window_size = max(int(len(component_indices) * intensity * 0.4), 5)
        start_pos = self.rng.randint(0, len(component_indices) - window_size - 1)

        dropout_start = component_indices[start_pos]
        dropout_end = component_indices[min(start_pos + window_size, len(component_indices) - 1)]

        # Remove logs in the window (simulate dropout)
        modified_logs = [
            log
            for i, log in enumerate(logs)
            if not (dropout_start <= i <= dropout_end and log.component == target_component)
        ]

        return modified_logs, {
            "anomaly_type": AnomalyType.SERVICE_DROPOUT.value,
            "component": target_component,
            "start_time": logs[dropout_start].timestamp,
            "end_time": logs[dropout_end].timestamp,
            "intensity": intensity,
            "lines_dropped": dropout_end - dropout_start + 1,
        }

    def _generate_error_message(self, component: str, timestamp: str) -> LogLine:
        """Generate a realistic error log message."""
        errors = [
            f"Connection timeout to {component}",
            f"Failed to process request in {component}",
            f"Exception in {component}: OutOfMemoryError",
            f"{component} service unavailable",
            f"Timeout waiting for response from {component}",
            f"Resource exhaustion in {component}",
        ]
        return LogLine(
            timestamp=timestamp,
            severity="ERROR",
            component=component,
            message=self.rng.choice(errors),
            raw_line=f"{timestamp} ERROR {component} {self.rng.choice(errors)}",
        )

    def _generate_memory_log(self, component: str, timestamp: str, memory_mb: int) -> LogLine:
        """Generate a memory-related log message."""
        gc_pause = memory_mb * 0.1 + self.rng.gauss(0, 10)
        return LogLine(
            timestamp=timestamp,
            severity="WARN",
            component=component,
            message=f"GC pause: {gc_pause:.1f}ms, Heap: {memory_mb}MB / 4096MB",
            raw_line=f"{timestamp} WARN {component} GC[ParNew]: {gc_pause:.1f}ms, heap: {memory_mb}->{memory_mb - 100}MB",
        )

    def _inject_cascade_failure(
        self, logs: List[LogLine], intensity: float
    ) -> Tuple[List[LogLine], Dict[str, Any]]:
        """
        Inject cascade failure anomaly - service A fails, causing B and C to fail in sequence.
        This is the hardest anomaly type requiring multi-hop reasoning.
        """
        if not logs:
            return logs, {}

        components = list(set(l.component for l in logs if l.component != "unknown"))
        if len(components) < 3:
            components = ["service_a", "service_b", "service_c"]

        # Select cascade chain
        num_cascade = min(3, len(components))
        cascade_chain = self.rng.sample(components, num_cascade)

        # Define cascade timing with delays between stages
        start_idx = len(logs) // 4
        end_idx = len(logs) * 3 // 4
        cascade_duration = end_idx - start_idx
        stage_duration = cascade_duration // (num_cascade + 1)

        modified_logs = list(logs)
        cascade_stages = []

        for stage, component in enumerate(cascade_chain):
            stage_start = start_idx + stage * stage_duration
            stage_end = stage_start + stage_duration

            # Inject errors for this stage
            for i in range(stage_start, min(stage_end, len(modified_logs))):
                if self.rng.random() < intensity * 0.7:
                    error_log = self._generate_cascade_error(
                        component, cascade_chain, stage, logs[i].timestamp
                    )
                    modified_logs.insert(i + 1, error_log)

            cascade_stages.append(
                {
                    "component": component,
                    "start_time": logs[stage_start].timestamp
                    if stage_start < len(logs)
                    else logs[0].timestamp,
                    "end_time": logs[min(stage_end, len(logs) - 1)].timestamp,
                }
            )

        return modified_logs, {
            "anomaly_type": AnomalyType.CASCADE_FAILURE.value,
            "component": cascade_chain[0],  # Primary component
            "cascade_chain": cascade_chain,
            "start_time": logs[start_idx].timestamp if start_idx < len(logs) else logs[0].timestamp,
            "end_time": logs[min(end_idx, len(logs) - 1)].timestamp,
            "intensity": intensity,
            "cascade_stages": cascade_stages,
        }

    def _generate_cascade_error(
        self, component: str, cascade_chain: List[str], stage: int, timestamp: str
    ) -> LogLine:
        """Generate cascade failure error message."""
        if stage == 0:
            messages = [
                f"Initial failure in {component}",
                f"{component} connection refused",
                f"Circuit breaker OPEN for {component}",
            ]
        else:
            upstream = cascade_chain[stage - 1]
            messages = [
                f"Cascaded failure from {upstream} to {component}",
                f"{component} timeout waiting for {upstream}",
                f"Dependency failure: {component} affected by {upstream} outage",
                f"{component} entering degraded mode due to {upstream} failure",
            ]

        severity = "ERROR" if stage == 0 else "WARN"
        message = self.rng.choice(messages)

        return LogLine(
            timestamp=timestamp,
            severity=severity,
            component=component,
            message=message,
            raw_line=f"{timestamp} {severity} {component} {message}",
        )

    def _inject_latency_degradation(
        self, logs: List[LogLine], intensity: float
    ) -> Tuple[List[LogLine], Dict[str, Any]]:
        """
        Inject latency degradation anomaly - gradually increasing response times.
        Requires correlating timing patterns across the log.
        """
        if not logs:
            return logs, {}

        components = list(set(l.component for l in logs if l.component != "unknown"))
        if not components:
            components = ["api_gateway"]

        target_component = self.rng.choice(components)

        # Define degradation window
        start_idx = len(logs) // 5
        end_idx = len(logs) * 4 // 5
        window_size = end_idx - start_idx

        modified_logs = []
        latency_values = []

        for i, log in enumerate(logs):
            modified_logs.append(log)

            if start_idx <= i < end_idx:
                progress = (i - start_idx) / window_size
                # Exponential latency growth
                base_latency = 50 + progress * 500 * intensity
                latency = base_latency + self.rng.gauss(0, base_latency * 0.1)
                latency_values.append(latency)

                latency_log = self._generate_latency_log(
                    target_component, log.timestamp, latency, intensity
                )
                modified_logs.append(latency_log)

        return modified_logs, {
            "anomaly_type": AnomalyType.LATENCY_DEGRADATION.value,
            "component": target_component,
            "start_time": logs[start_idx].timestamp if start_idx < len(logs) else logs[0].timestamp,
            "end_time": logs[min(end_idx, len(logs) - 1)].timestamp,
            "intensity": intensity,
            "peak_latency": max(latency_values) if latency_values else 0,
            "avg_latency": sum(latency_values) / len(latency_values) if latency_values else 0,
        }

    def _generate_latency_log(
        self, component: str, timestamp: str, latency_ms: float, intensity: float
    ) -> LogLine:
        """Generate latency-related log message."""
        severity = "WARN" if latency_ms < 300 else "ERROR"
        message = f"Request latency: {latency_ms:.1f}ms (threshold: 200ms)"

        return LogLine(
            timestamp=timestamp,
            severity=severity,
            component=component,
            message=message,
            raw_line=f"{timestamp} {severity} {component} [PERF] {message}",
        )

    def _inject_auth_anomaly(
        self, logs: List[LogLine], intensity: float
    ) -> Tuple[List[LogLine], Dict[str, Any]]:
        """
        Inject authentication anomaly - repeated failed login attempts.
        Requires identifying patterns in auth logs.
        """
        if not logs:
            return logs, {}

        # Auth anomalies typically come from auth_service or similar
        components = list(set(l.component for l in logs if l.component != "unknown"))
        auth_component = self.rng.choice(
            [c for c in components if "auth" in c.lower() or "login" in c.lower()]
            or ["auth_service"]
        )

        # Define anomaly window
        start_idx = len(logs) // 4
        end_idx = len(logs) * 3 // 4
        window_size = end_idx - start_idx

        # Generate suspicious source
        attacker_ip = f"192.168.{self.rng.randint(1, 254)}.{self.rng.randint(1, 254)}"
        target_user = self.rng.choice(["admin", "root", "deploy", "service_account", "user_1"])

        modified_logs = []
        failed_attempts = 0

        for i, log in enumerate(logs):
            modified_logs.append(log)

            if start_idx <= i < end_idx:
                # Inject failed auth attempts at irregular intervals
                if self.rng.random() < intensity * 0.3:
                    failed_attempts += 1
                    auth_log = self._generate_auth_failure(
                        auth_component, log.timestamp, attacker_ip, target_user
                    )
                    modified_logs.append(auth_log)

        return modified_logs, {
            "anomaly_type": AnomalyType.AUTH_ANOMALY.value,
            "component": auth_component,
            "start_time": logs[start_idx].timestamp if start_idx < len(logs) else logs[0].timestamp,
            "end_time": logs[min(end_idx, len(logs) - 1)].timestamp,
            "intensity": intensity,
            "source_ip": attacker_ip,
            "target_user": target_user,
            "failed_attempts": failed_attempts,
        }

    def _generate_auth_failure(
        self, component: str, timestamp: str, source_ip: str, target_user: str
    ) -> LogLine:
        """Generate authentication failure log message."""
        messages = [
            f"Authentication failed for {target_user} from {source_ip}",
            f"Invalid credentials for {target_user} from {source_ip}",
            f"Login attempt blocked for {target_user} from {source_ip}",
            f"Brute force detection triggered for {source_ip}",
        ]
        message = self.rng.choice(messages)

        return LogLine(
            timestamp=timestamp,
            severity="WARN",
            component=component,
            message=message,
            raw_line=f"{timestamp} WARN {component} [AUTH] {message}",
        )


def generate_synthetic_log(
    num_lines: int = 1000,
    num_components: int = 5,
    seed: Optional[int] = None,
) -> Tuple[List[LogLine], Dict[str, Any]]:
    """
    Generate synthetic log data with ground truth.

    Args:
        num_lines: Number of log lines to generate
        num_components: Number of distinct components
        seed: Random seed

    Returns:
        Tuple of (generated logs, metadata)
    """
    rng = random.Random(seed)

    components = [f"service_{chr(97 + i)}" for i in range(num_components)]
    severities = ["DEBUG", "INFO", "INFO", "INFO", "WARN"]

    base_time = datetime.now() - timedelta(hours=1)

    logs = []
    for i in range(num_lines):
        timestamp = base_time + timedelta(seconds=i * 3.6)  # ~1000 lines per hour
        severity = rng.choice(severities)
        component = rng.choice(components)

        messages = [
            f"Processing request {i}",
            f"Connection established to {rng.choice(components)}",
            f"Checkpoint reached at block {i % 100}",
            f"Heartbeat sent to coordinator",
            f"Data chunk {i} transferred successfully",
        ]

        logs.append(
            LogLine(
                timestamp=timestamp.isoformat(),
                severity=severity,
                component=component,
                message=rng.choice(messages),
                raw_line=f"{timestamp.isoformat()} {severity} {component} {rng.choice(messages)}",
            )
        )

    return logs, {"num_lines": num_lines, "components": components}
