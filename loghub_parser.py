"""
LogHub Parser for Real Log Data.

This module provides parsers for real LogHub datasets including HDFS, BGL,
OpenStack, and Apache logs. These parsers handle authentic log formats from
production systems and are used for creating realistic evaluation scenarios.
"""
import re
import os
import sys
import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field

# Support both package and direct execution modes
if __package__:
    from .models import AnomalyType, DifficultyLevel, LogLine
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import AnomalyType, DifficultyLevel, LogLine


class LogHubMetadata(NamedTuple):
    """Metadata for parsed LogHub data."""
    source: str  # HDFS, BGL, OpenStack, Apache
    time_range: Tuple[datetime, datetime]
    components: List[str]
    severities: List[str]
    total_lines: int
    has_labels: bool
    labels: Optional[Dict[int, str]] = None  # line_index -> label


class HDFSLogParser:
    """
    Parser for HDFS (Hadoop Distributed File System) logs.

    HDFS log format:
    1116 08-19 15:41:44 1488 INFO BlockManager: Compression is off

    Format: <BlockId> <Date> <Time> <Pid> <Level> <Source>: <Content>
    """

    # HDFS log pattern
    PATTERN = re.compile(
        r'^(?P<block_id>\d+)\s+'
        r'(?P<date>\d{2}-\d{2})\s+'
        r'(?P<time>\d{2}:\d{2}:\d{2})\s+'
        r'(?P<pid>\d+)\s+'
        r'(?P<level>[\w]+)\s+'
        r'(?P<source>[\w\.]+):\s*'
        r'(?P<content>.*)$'
    )

    # Block-related components
    BLOCK_COMPONENTS = [
        "BlockManager", "DataNode", "NameNode", "FSNamesystem",
        "Replication", "Heartbeat", "BlockReport", "BlockPool"
    ]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.base_year = 2009  # HDFS_v1 is from 2009

    def parse_line(self, line: str, line_number: int = 0) -> Optional[LogLine]:
        """Parse a single HDFS log line."""
        line = line.strip()
        if not line:
            return None

        match = self.PATTERN.match(line)
        if match:
            month, day = match.group('date').split('-')
            timestamp = f"2009-{month}-{day}T{match.group('time')}:00"

            return LogLine(
                timestamp=timestamp,
                severity=self._normalize_severity(match.group('level')),
                component=match.group('source'),
                message=match.group('content'),
                raw_line=line
            )

        # Fallback for unparseable lines
        return self._parse_fallback(line)

    def _normalize_severity(self, level: str) -> str:
        """Normalize HDFS log levels to standard levels."""
        level_upper = level.upper()
        if level_upper in ('FATAL', 'ERROR'):
            return 'ERROR'
        elif level_upper == 'WARN':
            return 'WARN'
        elif level_upper in ('INFO', 'DEBUG', 'TRACE'):
            return 'INFO'
        return 'INFO'

    def _parse_fallback(self, line: str) -> Optional[LogLine]:
        """Fallback parser for non-standard HDFS lines."""
        severity = 'INFO'
        if 'ERROR' in line:
            severity = 'ERROR'
        elif 'WARN' in line:
            severity = 'WARN'
        elif 'FATAL' in line:
            severity = 'ERROR'

        # Extract timestamp
        timestamp_match = re.search(r'\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line)
        timestamp = timestamp_match.group(0) if timestamp_match else datetime.now().isoformat()
        timestamp = f"2009-{timestamp.replace(' ', 'T')}"

        return LogLine(
            timestamp=timestamp,
            severity=severity,
            component="hdfs_unknown",
            message=line,
            raw_line=line
        )

    def parse_file(self, filepath: str, max_lines: Optional[int] = None) -> Tuple[List[LogLine], LogHubMetadata]:
        """Parse an entire HDFS log file."""
        lines = []
        timestamps = []
        components = set()
        severities = set()

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                parsed = self.parse_line(line, i)
                if parsed:
                    lines.append(parsed)
                    timestamps.append(parsed.timestamp)
                    components.add(parsed.component)
                    severities.add(parsed.severity)

        time_range = (
            datetime.fromisoformat(min(timestamps)) if timestamps else datetime(2009, 1, 1),
            datetime.fromisoformat(max(timestamps)) if timestamps else datetime(2009, 1, 1)
        )

        return lines, LogHubMetadata(
            source="HDFS",
            time_range=time_range,
            components=list(components),
            severities=list(severities),
            total_lines=len(lines),
            has_labels=False
        )


class BGLLogParser:
    """
    Parser for BGL (Blue Gene/L) supercomputer logs.

    BGL log format:
    2006-07-28-21.36.23.281500 R813-M0-US2-C0-J01 0 [error] node_recovery:...

    Format: <Timestamp> <NodeId> <JobId> [Severity] <Message>
    """

    PATTERN = re.compile(
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d{6})\s+'
        r'(?P<node>\S+)\s+'
        r'(?P<job>\d+|N/A)\s+'
        r'\[(?P<severity>[\w]+)\]\s*'
        r'(?P<message>.*)$'
    )

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def parse_line(self, line: str, line_number: int = 0) -> Optional[LogLine]:
        """Parse a single BGL log line."""
        line = line.strip()
        if not line:
            return None

        # Skip comment lines
        if line.startswith('##'):
            return None

        match = self.PATTERN.match(line)
        if match:
            # Convert BGL timestamp format
            ts = match.group('timestamp')
            dt = datetime.strptime(ts, "%Y-%m-%d-%H.%M.%S.%f")

            return LogLine(
                timestamp=dt.isoformat(),
                severity=self._normalize_severity(match.group('severity')),
                component=match.group('node'),
                message=match.group('message'),
                raw_line=line
            )

        return self._parse_fallback(line)

    def _normalize_severity(self, severity: str) -> str:
        """Normalize BGL severity to standard levels."""
        sev_upper = severity.upper()
        if sev_upper == 'ERR' or 'ERROR' in sev_upper:
            return 'ERROR'
        elif 'WARNING' in sev_upper or 'WARN' in sev_upper:
            return 'WARN'
        elif 'FATAL' in sev_upper:
            return 'ERROR'
        return 'INFO'

    def _parse_fallback(self, line: str) -> Optional[LogLine]:
        """Fallback parser for non-standard BGL lines."""
        severity = 'INFO'
        if 'error' in line.lower():
            severity = 'ERROR'
        elif 'warning' in line.lower():
            severity = 'WARN'

        timestamp_match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}', line)
        timestamp = timestamp_match.group(0) if timestamp_match else datetime.now().isoformat()
        timestamp = timestamp.replace('-', 'T').replace('.', ':', 2).replace('.', ':').replace(':', 'T', 1)

        node_match = re.search(r'[A-Z]\d{3}-[A-Z]\d-[A-Z][A-Z]\d-[A-Z]\d\d', line)
        node = node_match.group(0) if node_match else 'unknown'

        return LogLine(
            timestamp=timestamp,
            severity=severity,
            component=node,
            message=line,
            raw_line=line
        )

    def parse_file(self, filepath: str, max_lines: Optional[int] = None) -> Tuple[List[LogLine], LogHubMetadata]:
        """Parse an entire BGL log file."""
        lines = []
        timestamps = []
        components = set()
        severities = set()

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                parsed = self.parse_line(line, i)
                if parsed:
                    lines.append(parsed)
                    timestamps.append(parsed.timestamp)
                    components.add(parsed.component)
                    severities.add(parsed.severity)

        time_range = (
            datetime.fromisoformat(min(timestamps)) if timestamps else datetime(2006, 1, 1),
            datetime.fromisoformat(max(timestamps)) if timestamps else datetime(2006, 1, 1)
        )

        return lines, LogHubMetadata(
            source="BGL",
            time_range=time_range,
            components=list(components),
            severities=list(severities),
            total_lines=len(lines),
            has_labels=False
        )


class OpenStackLogParser:
    """
    Parser for OpenStack logs.

    OpenStack log format (Zuul/CI style):
    2016-08-02 18:51:17.958 3089 INFO oslo.service.periodic_task [-] ...

    Format: <Timestamp> <Pid> <Level> <Source> <Content>
    """

    PATTERN = re.compile(
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+'
        r'(?P<pid>\d+)\s+'
        r'(?P<level>\w+)\s+'
        r'(?P<source>\S+)\s+'
        r'\[(?P<extra>.*?)\]\s*'
        r'(?P<message>.*)$'
    )

    # OpenStack service components
    OPENSTACK_COMPONENTS = [
        "nova", "neutron", "glance", "cinder", "keystone", "swift",
        "horizon", "ceilometer", "heat", "ironic", "zaqar", "oslo"
    ]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def parse_line(self, line: str, line_number: int = 0) -> Optional[LogLine]:
        """Parse a single OpenStack log line."""
        line = line.strip()
        if not line:
            return None

        match = self.PATTERN.match(line)
        if match:
            timestamp = match.group('timestamp').replace(' ', 'T')

            return LogLine(
                timestamp=timestamp,
                severity=self._normalize_severity(match.group('level')),
                component=match.group('source'),
                message=match.group('message'),
                raw_line=line
            )

        return self._parse_fallback(line)

    def _normalize_severity(self, level: str) -> str:
        """Normalize OpenStack log levels to standard levels."""
        level_upper = level.upper()
        if level_upper in ('ERROR', 'ERR'):
            return 'ERROR'
        elif level_upper in ('WARNING', 'WARN'):
            return 'WARN'
        elif level_upper == 'CRITICAL':
            return 'ERROR'
        elif level_upper in ('DEBUG', 'TRACE'):
            return 'DEBUG'
        return 'INFO'

    def _parse_fallback(self, line: str) -> Optional[LogLine]:
        """Fallback parser for non-standard OpenStack lines."""
        severity = 'INFO'
        if 'ERROR' in line or 'error' in line.lower():
            severity = 'ERROR'
        elif 'WARNING' in line or 'warn' in line.lower():
            severity = 'WARN'

        timestamp_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line)
        timestamp = timestamp_match.group(0) if timestamp_match else datetime.now().isoformat()
        timestamp = timestamp.replace(' ', 'T')

        return LogLine(
            timestamp=timestamp,
            severity=severity,
            component="openstack",
            message=line,
            raw_line=line
        )

    def parse_file(self, filepath: str, max_lines: Optional[int] = None) -> Tuple[List[LogLine], LogHubMetadata]:
        """Parse an entire OpenStack log file."""
        lines = []
        timestamps = []
        components = set()
        severities = set()

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                parsed = self.parse_line(line, i)
                if parsed:
                    lines.append(parsed)
                    timestamps.append(parsed.timestamp)
                    components.add(parsed.component)
                    severities.add(parsed.severity)

        time_range = (
            datetime.fromisoformat(min(timestamps)) if timestamps else datetime(2016, 1, 1),
            datetime.fromisoformat(max(timestamps)) if timestamps else datetime(2016, 1, 1)
        )

        return lines, LogHubMetadata(
            source="OpenStack",
            time_range=time_range,
            components=list(components),
            severities=list(severities),
            total_lines=len(lines),
            has_labels=False
        )


class ApacheLogParser:
    """
    Parser for Apache web server error logs.

    Apache error log format:
    [Wed Oct 11 14:32:52 2000] [error] [client 1.2.3.4] File does not exist: /var/www/html/favicon.ico

    Format: [<Timestamp>] [Level] [client <IP>] <Message>
    """

    PATTERN = re.compile(
        r'^\[(?P<timestamp>[^\]]+)\]\s*'
        r'\[(?P<level>\w+)\]\s*'
        r'(?:\[client (?P<client>[^\]]+)\])?\s*'
        r'(?P<message>.*)$'
    )

    MONTH_MAP = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def parse_line(self, line: str, line_number: int = 0) -> Optional[LogLine]:
        """Parse a single Apache error log line."""
        line = line.strip()
        if not line:
            return None

        match = self.PATTERN.match(line)
        if match:
            timestamp_str = match.group('timestamp')
            timestamp = self._parse_apache_timestamp(timestamp_str)

            client = match.group('client') or 'unknown'

            return LogLine(
                timestamp=timestamp,
                severity=self._normalize_severity(match.group('level')),
                component=f"apache_{client}",
                message=match.group('message'),
                raw_line=line
            )

        return self._parse_fallback(line)

    def _parse_apache_timestamp(self, ts: str) -> str:
        """Convert Apache timestamp to ISO format."""
        # Format: "Wed Oct 11 14:32:52 2000"
        parts = ts.split()
        if len(parts) >= 5:
            day, month, day_num, time, year = parts[:5]
            month_num = self.MONTH_MAP.get(month, '01')
            return f"{year}-{month_num}-{day_num}T{time}"
        return datetime.now().isoformat()

    def _normalize_severity(self, level: str) -> str:
        """Normalize Apache log levels to standard levels."""
        level_upper = level.upper()
        if level_upper in ('ERROR', 'CRIT', 'ALERT', 'EMERG'):
            return 'ERROR'
        elif level_upper in ('WARNING', 'WARN'):
            return 'WARN'
        elif level_upper == 'NOTICE':
            return 'INFO'
        return 'INFO'

    def _parse_fallback(self, line: str) -> Optional[LogLine]:
        """Fallback parser for non-standard Apache lines."""
        severity = 'INFO'
        if 'error' in line.lower():
            severity = 'ERROR'
        elif 'warn' in line.lower():
            severity = 'WARN'

        timestamp_match = re.search(r'\d{4}-\d{2}-\d{2}|\[\w+ \w+ \d+ \d{2}:\d{2}:\d{2}', line)
        timestamp = timestamp_match.group(0) if timestamp_match else datetime.now().isoformat()
        if timestamp.startswith('['):
            timestamp = timestamp[1:]

        return LogLine(
            timestamp=timestamp,
            severity=severity,
            component="apache",
            message=line,
            raw_line=line
        )

    def parse_file(self, filepath: str, max_lines: Optional[int] = None) -> Tuple[List[LogLine], LogHubMetadata]:
        """Parse an entire Apache log file."""
        lines = []
        timestamps = []
        components = set()
        severities = set()

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                parsed = self.parse_line(line, i)
                if parsed:
                    lines.append(parsed)
                    timestamps.append(parsed.timestamp)
                    components.add(parsed.component)
                    severities.add(parsed.severity)

        time_range = (
            datetime.fromisoformat(min(timestamps)) if timestamps else datetime(2016, 1, 1),
            datetime.fromisoformat(max(timestamps)) if timestamps else datetime(2016, 1, 1)
        )

        return lines, LogHubMetadata(
            source="Apache",
            time_range=time_range,
            components=list(components),
            severities=list(severities),
            total_lines=len(lines),
            has_labels=False
        )


class LogHubFactory:
    """
    Factory for creating LogHub parsers based on log source type.

    Supports:
    - HDFS: Hadoop Distributed File System logs
    - BGL: Blue Gene/L supercomputer logs
    - OpenStack: OpenStack infrastructure logs
    - Apache: Apache web server error logs
    """

    PARSERS = {
        'HDFS': HDFSLogParser,
        'BGL': BGLLogParser,
        'OpenStack': OpenStackLogParser,
        'Apache': ApacheLogParser,
    }

    @classmethod
    def get_parser(cls, source: str, seed: Optional[int] = None):
        """
        Get a parser for the specified log source.

        Args:
            source: Log source name (HDFS, BGL, OpenStack, Apache)
            seed: Optional random seed

        Returns:
            Appropriate parser instance
        """
        source_upper = source.upper()
        parser_class = cls.PARSERS.get(source_upper)
        if parser_class is None:
            raise ValueError(f"Unknown log source: {source}. Available: {list(cls.PARSERS.keys())}")
        return parser_class(seed=seed)

    @classmethod
    def parse_file(cls, filepath: str, source: Optional[str] = None, max_lines: Optional[int] = None) -> Tuple[List[LogLine], LogHubMetadata]:
        """
        Parse a log file, auto-detecting the source if not specified.

        Args:
            filepath: Path to the log file
            source: Optional source type hint
            max_lines: Optional maximum lines to parse

        Returns:
            Tuple of (parsed lines, metadata)
        """
        if source is None:
            source = cls._detect_source(filepath)

        parser = cls.get_parser(source)
        return parser.parse_file(filepath, max_lines=max_lines)

    @classmethod
    def _detect_source(cls, filepath: str) -> str:
        """Auto-detect log source from filepath or content."""
        filepath_lower = filepath.lower()

        # Detect from filename
        if 'hdfs' in filepath_lower:
            return 'HDFS'
        elif 'bgl' in filepath_lower:
            return 'BGL'
        elif 'openstack' in filepath_lower:
            return 'OpenStack'
        elif 'apache' in filepath_lower:
            return 'Apache'

        # Try to detect from content
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                sample = f.read(1000)

                if re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', sample):
                    return 'HDFS'
                elif re.search(r'\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}', sample):
                    return 'BGL'
                elif re.search(r'\[\w+ \w+ \d+ \d{2}:\d{2}:\d{2}', sample):
                    return 'Apache'
                elif re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+', sample):
                    return 'OpenStack'
        except:
            pass

        # Default to HDFS
        return 'HDFS'

    @classmethod
    def list_sources(cls) -> List[str]:
        """List all available log sources."""
        return list(cls.PARSERS.keys())


class LogHubSampler:
    """
    Samples and slices LogHub data for environment episodes.

    This class handles:
    - Random sampling of log segments
    - Segment size based on difficulty
    - Ground truth extraction for evaluation
    """

    DIFFICULTY_SIZES = {
        DifficultyLevel.EASY: (400, 600),
        DifficultyLevel.MEDIUM: (800, 1200),
        DifficultyLevel.HARD: (1500, 2500),
    }

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def sample_segment(
        self,
        logs: List[LogLine],
        difficulty: DifficultyLevel,
        anomaly_region: Optional[Tuple[int, int]] = None
    ) -> Tuple[List[LogLine], Dict[str, Any]]:
        """
        Sample a segment of logs for an episode.

        Args:
            logs: Full log list
            difficulty: Target difficulty
            anomaly_region: Optional (start, end) indices for anomaly location

        Returns:
            Tuple of (sampled logs, metadata including ground truth)
        """
        if not logs:
            return [], {}

        # Determine segment size
        min_size, max_size = self.DIFFICULTY_SIZES.get(difficulty, (500, 1000))
        segment_size = self.rng.randint(min_size, max_size)

        # Clamp to available logs
        segment_size = min(segment_size, len(logs))

        # Determine start position
        if anomaly_region:
            # Ensure anomaly is in the segment
            anom_start, anom_end = anomaly_region

            # Anomaly should be in middle 60% of segment
            buffer = segment_size * 0.2
            min_start = max(0, int(anom_end - segment_size + buffer))
            max_start = min(len(logs) - segment_size, int(anom_start - buffer))

            if max_start > min_start:
                start_idx = self.rng.randint(min_start, max_start)
            else:
                start_idx = min_start
        else:
            # Random segment from middle 80%
            margin = len(logs) * 0.1
            min_start = int(margin)
            max_start = int(len(logs) - segment_size - margin)
            start_idx = self.rng.randint(min_start, max_start) if max_start > min_start else min_start

        # Extract segment
        segment = logs[start_idx:start_idx + segment_size]

        # Generate ground truth
        ground_truth = {
            "source_idx": start_idx,
            "segment_size": len(segment),
            "total_size": len(logs),
        }

        return segment, ground_truth

    def create_eval_sample(
        self,
        logs: List[LogLine],
        metadata: LogHubMetadata,
        difficulty: DifficultyLevel
    ) -> Tuple[List[LogLine], Dict[str, Any]]:
        """
        Create an evaluation sample with ground truth from labeled data.

        Args:
            logs: Full log list
            metadata: LogHub metadata
            difficulty: Target difficulty

        Returns:
            Tuple of (sampled logs, ground truth)
        """
        segment, ground_truth = self.sample_segment(logs, difficulty)

        # Add metadata
        ground_truth.update({
            "log_source": metadata.source,
            "difficulty": difficulty.value,
        })

        return segment, ground_truth


def load_loghub_sample(
    source: str,
    data_dir: str = "./data/loghub",
    max_lines: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[List[LogLine], LogHubMetadata]:
    """
    Load a sample from LogHub data.

    Args:
        source: Log source (HDFS, BGL, OpenStack, Apache)
        data_dir: Directory containing LogHub data
        max_lines: Optional maximum lines to load
        seed: Optional random seed

    Returns:
        Tuple of (parsed logs, metadata)
    """
    # Map sources to filenames
    filenames = {
        'HDFS': 'hdfs.log',
        'BGL': 'bgl.log',
        'OpenStack': 'openstack.log',
        'Apache': 'apache_error.log',
    }

    filename = filenames.get(source.upper())
    if filename is None:
        raise ValueError(f"Unknown source: {source}")

    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"LogHub file not found: {filepath}")

    return LogHubFactory.parse_file(filepath, source=source, max_lines=max_lines)
