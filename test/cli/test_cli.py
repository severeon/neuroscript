import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))  # Add src to path

from cli.main import main
from core.graph_validator import ValidationResult
from core.container_runtime import ExecutionResult


@patch('cli.main.BlockRegistry')
@patch('cli.main.GraphLoader')
@patch('cli.main.GraphValidator')
def test_validate_success(mock_validator, mock_loader, mock_registry, capsys):
    mock_graph = MagicMock()
    mock_loader.return_value.load.return_value = mock_graph
    mock_validator.return_value.validate.return_value = ValidationResult(valid=True)

    with patch.object(sys, 'argv', ['neuroscript', 'validate', 'test.yaml']):
        main()

    captured = capsys.readouterr()
    assert '✅ Architecture is valid!' in captured.out


@patch('cli.main.BlockRegistry')
@patch('cli.main.GraphLoader')
@patch('cli.main.GraphValidator')
def test_validate_failure(mock_validator, mock_loader, mock_registry, capsys):
    mock_graph = MagicMock()
    mock_loader.return_value.load.return_value = mock_graph
    mock_result = ValidationResult(valid=False, errors=[MagicMock(message='Test error')])
    mock_validator.return_value.validate.return_value = mock_result

    with patch.object(sys, 'argv', ['neuroscript', 'validate', 'test.yaml']):
        with pytest.raises(SystemExit):
            main()

    captured = capsys.readouterr()
    assert '❌ Validation failed:' in captured.out
    assert 'Test error' in captured.out


@patch('cli.main.BlockRegistry')
@patch('cli.main.GraphLoader')
@patch('cli.main.GraphValidator')
@patch('cli.main.CompilationEngine')
@patch('pathlib.Path.write_text')
def test_compile_success(mock_write, mock_engine, mock_validator, mock_loader, mock_registry, capsys, tmp_path):
    mock_graph = MagicMock()
    mock_loader.return_value.load.return_value = mock_graph
    mock_validator.return_value.validate.return_value = ValidationResult(valid=True)
    mock_engine.return_value.compile.return_value = 'code'

    output_file = tmp_path / 'model.py'
    with patch.object(sys, 'argv', ['neuroscript', 'compile', 'test.yaml', '--output', str(output_file)]):
        main()

    captured = capsys.readouterr()
    assert f'✅ Compiled to {output_file}' in captured.out


@patch('cli.main.ContainerRuntime')
def test_run_success(mock_runtime, capsys):
    mock_result = ExecutionResult(exit_code=0, stdout='Success output', stderr="", container_id="container-id", oom_killed=False, execution_time_seconds=1)
    mock_runtime.return_value.execute.return_value = mock_result

    with patch.object(sys, 'argv', ['neuroscript', 'run', 'model.py']):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert 'Execution Summary' in captured.out


def test_help(capsys):
    with patch.object(sys, 'argv', ['neuroscript', '--help']):
        with pytest.raises(SystemExit):
            main()

    captured = capsys.readouterr()
    assert 'Validate architecture graph' in captured.out
