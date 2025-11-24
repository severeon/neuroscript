"""Tests for monitoring module."""

import json
import time
from io import StringIO
from contextlib import redirect_stdout

import pytest

from core.monitoring import (
    ExecutionMetrics,
    MonitorEvent,
    ResourceSample,
    emit_event,
    format_event,
    format_execution_summary,
    parse_event_line,
    timing_context
)


class TestMonitorEvent:
    """Tests for MonitorEvent dataclass."""

    def test_to_json(self):
        """Test JSON serialization."""
        event = MonitorEvent(
            type='block_forward',
            timestamp=1234567890.0,
            data={'block_name': 'encoder', 'time_ms': 15.3}
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed['type'] == 'block_forward'
        assert parsed['timestamp'] == 1234567890.0
        assert parsed['data']['block_name'] == 'encoder'
        assert parsed['data']['time_ms'] == 15.3

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = json.dumps({
            'type': 'epoch_start',
            'timestamp': 1234567890.0,
            'data': {'epoch': 1, 'total_epochs': 10}
        })

        event = MonitorEvent.from_json(json_str)

        assert event.type == 'epoch_start'
        assert event.timestamp == 1234567890.0
        assert event.data['epoch'] == 1
        assert event.data['total_epochs'] == 10


class TestResourceSample:
    """Tests for ResourceSample dataclass."""

    def test_creation_cpu_only(self):
        """Test creating CPU-only resource sample."""
        sample = ResourceSample(
            timestamp=time.time(),
            cpu_percent=75.5,
            memory_mb=1024.0,
            memory_percent=50.0
        )

        assert sample.cpu_percent == 75.5
        assert sample.memory_mb == 1024.0
        assert sample.memory_percent == 50.0
        assert sample.gpu_utilization is None
        assert sample.gpu_memory_mb is None

    def test_creation_with_gpu(self):
        """Test creating resource sample with GPU stats."""
        sample = ResourceSample(
            timestamp=time.time(),
            cpu_percent=75.5,
            memory_mb=1024.0,
            memory_percent=50.0,
            gpu_utilization=95.0,
            gpu_memory_mb=2048.0
        )

        assert sample.gpu_utilization == 95.0
        assert sample.gpu_memory_mb == 2048.0


class TestEmitEvent:
    """Tests for emit_event function."""

    def test_emit_event(self):
        """Test event emission to stdout."""
        output = StringIO()

        with redirect_stdout(output):
            emit_event('block_forward', {
                'block_name': 'encoder',
                'input_shape': [32, 512],
                'output_shape': [32, 256],
                'time_ms': 15.3
            })

        output_str = output.getvalue()
        assert output_str.startswith('NEUROSCRIPT_EVENT:')

        # Parse the JSON part
        json_str = output_str.replace('NEUROSCRIPT_EVENT:', '').strip()
        event_data = json.loads(json_str)

        assert event_data['type'] == 'block_forward'
        assert event_data['data']['block_name'] == 'encoder'
        assert event_data['data']['time_ms'] == 15.3


class TestFormatEvent:
    """Tests for format_event function."""

    def test_format_execution_start(self):
        """Test formatting execution start event."""
        event = MonitorEvent(
            type='execution_start',
            timestamp=1234567890.0,
            data={'mode': 'inference', 'input_shape': [1, 512]}
        )

        formatted = format_event(event)
        assert 'Starting inference' in formatted
        assert '[1, 512]' in formatted

    def test_format_block_forward(self):
        """Test formatting block forward event."""
        event = MonitorEvent(
            type='block_forward',
            timestamp=1234567890.0,
            data={
                'block_name': 'encoder',
                'input_shape': [32, 512],
                'output_shape': [32, 256],
                'time_ms': 15.3
            }
        )

        formatted = format_event(event)
        assert 'encoder' in formatted
        assert '(32, 512)' in formatted
        assert '(32, 256)' in formatted
        assert '15.3ms' in formatted

    def test_format_epoch_start(self):
        """Test formatting epoch start event."""
        event = MonitorEvent(
            type='epoch_start',
            timestamp=1234567890.0,
            data={'epoch': 1, 'total_epochs': 10}
        )

        formatted = format_event(event)
        assert 'Epoch 1/10' in formatted
        assert 'started' in formatted

    def test_format_batch_complete(self):
        """Test formatting batch complete event."""
        event = MonitorEvent(
            type='batch_complete',
            timestamp=1234567890.0,
            data={
                'epoch': 1,
                'batch': 5,
                'total_batches': 100,
                'loss': 0.452,
                'metrics': {'accuracy': 0.873}
            }
        )

        formatted = format_event(event)
        assert 'Epoch 1' in formatted
        assert 'Batch 5/100' in formatted
        assert '0.452' in formatted
        assert 'Accuracy: 0.873' in formatted

    def test_format_epoch_complete(self):
        """Test formatting epoch complete event."""
        event = MonitorEvent(
            type='epoch_complete',
            timestamp=1234567890.0,
            data={
                'epoch': 1,
                'total_epochs': 10,
                'train_loss': 0.234,
                'val_loss': 0.267,
                'metrics': {'accuracy': 0.912}
            }
        )

        formatted = format_event(event)
        assert 'Epoch 1/10 complete' in formatted
        assert '0.234' in formatted
        assert '0.267' in formatted
        assert 'Accuracy: 0.912' in formatted

    def test_format_execution_end(self):
        """Test formatting execution end event."""
        event = MonitorEvent(
            type='execution_end',
            timestamp=1234567890.0,
            data={'success': True, 'total_time_seconds': 45.3}
        )

        formatted = format_event(event)
        assert 'completed successfully' in formatted
        assert '45.3' in formatted


class TestParseEventLine:
    """Tests for parse_event_line function."""

    def test_parse_valid_event(self):
        """Test parsing valid event line."""
        event_data = {
            'type': 'block_forward',
            'timestamp': 1234567890.0,
            'data': {'block_name': 'encoder', 'time_ms': 15.3}
        }
        line = f"NEUROSCRIPT_EVENT:{json.dumps(event_data)}"

        event = parse_event_line(line)

        assert event is not None
        assert event.type == 'block_forward'
        assert event.timestamp == 1234567890.0
        assert event.data['block_name'] == 'encoder'

    def test_parse_non_event_line(self):
        """Test parsing non-event line."""
        line = "This is just a regular output line"

        event = parse_event_line(line)

        assert event is None

    def test_parse_invalid_json(self):
        """Test parsing line with invalid JSON."""
        line = "NEUROSCRIPT_EVENT:{invalid json}"

        event = parse_event_line(line)

        assert event is None


class TestTimingContext:
    """Tests for timing_context manager."""

    def test_timing_context_basic(self):
        """Test basic timing context functionality."""
        output = StringIO()

        with redirect_stdout(output):
            with timing_context('test_block', emit_events=True) as ctx:
                ctx['input_shape'] = [32, 512]
                time.sleep(0.01)  # Simulate some work
                ctx['output_shape'] = [32, 256]

        output_str = output.getvalue()
        assert 'NEUROSCRIPT_EVENT:' in output_str

        # Parse event
        json_str = output_str.replace('NEUROSCRIPT_EVENT:', '').strip()
        event_data = json.loads(json_str)

        assert event_data['type'] == 'block_forward'
        assert event_data['data']['block_name'] == 'test_block'
        assert event_data['data']['input_shape'] == [32, 512]
        assert event_data['data']['output_shape'] == [32, 256]
        assert event_data['data']['time_ms'] > 0

    def test_timing_context_no_emit(self):
        """Test timing context with events disabled."""
        output = StringIO()

        with redirect_stdout(output):
            with timing_context('test_block', emit_events=False) as ctx:
                ctx['input_shape'] = [32, 512]
                ctx['output_shape'] = [32, 256]

        output_str = output.getvalue()
        assert 'NEUROSCRIPT_EVENT:' not in output_str


class TestFormatExecutionSummary:
    """Tests for format_execution_summary function."""

    def test_format_summary_cpu_only(self):
        """Test formatting summary for CPU-only execution."""
        metrics = ExecutionMetrics(
            total_time_seconds=45.3,
            peak_memory_mb=1024.5,
            peak_cpu_percent=87.3
        )

        summary = format_execution_summary(metrics)

        assert 'Execution Summary' in summary
        assert '45.3s' in summary
        assert '1024.5 MB' in summary
        assert '87.3%' in summary

    def test_format_summary_with_gpu(self):
        """Test formatting summary with GPU stats."""
        metrics = ExecutionMetrics(
            total_time_seconds=45.3,
            peak_memory_mb=1024.5,
            peak_cpu_percent=87.3,
            peak_gpu_utilization=95.2,
            peak_gpu_memory_mb=2048.0
        )

        summary = format_execution_summary(metrics)

        assert '95.2%' in summary
        assert '2048.0 MB' in summary

    def test_format_summary_with_training_metrics(self):
        """Test formatting summary with training metrics."""
        metrics = ExecutionMetrics(
            total_time_seconds=45.3,
            peak_memory_mb=1024.5,
            peak_cpu_percent=87.3,
            final_train_loss=0.234,
            final_val_loss=0.267,
            final_train_accuracy=0.912,
            final_val_accuracy=0.889
        )

        summary = format_execution_summary(metrics)

        assert 'Final Metrics' in summary
        assert '0.234' in summary
        assert '0.267' in summary
        assert '0.912' in summary
        assert '0.889' in summary

    def test_format_summary_with_layer_timings(self):
        """Test formatting summary with layer timings."""
        metrics = ExecutionMetrics(
            total_time_seconds=45.3,
            peak_memory_mb=1024.5,
            peak_cpu_percent=87.3,
            layer_timings={
                'encoder': 12.5,
                'decoder': 18.3,
                'output': 5.7
            }
        )

        summary = format_execution_summary(metrics)

        assert 'Layer Timings' in summary
        assert 'encoder' in summary
        assert '12.5ms' in summary
        assert 'decoder' in summary
        assert '18.3ms' in summary
