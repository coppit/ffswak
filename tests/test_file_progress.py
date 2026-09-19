"""Exercise the reporter through real pytest runs, including failure summaries."""
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


@pytest.mark.parametrize('workers', ['0', '2'])
def test_file_progress_preserves_results_and_failure_details(tmp_path, workers):
    (tmp_path / 'pytest.ini').write_text('[pytest]\n')
    (tmp_path / 'test_first.py').write_text('''
import pytest

def test_pass():
    pass

def test_fail():
    assert False, "failure detail preserved"

@pytest.mark.skip(reason="intentional skip")
def test_skip():
    pass
''')
    (tmp_path / 'test_second.py').write_text('''
import pytest

@pytest.mark.xfail(reason="known failure")
def test_xfail():
    assert False

@pytest.fixture
def broken():
    raise RuntimeError("setup detail preserved")

def test_error(broken):
    pass
''')
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-n', workers, '-p', 'file_progress', '--file-progress=on',
         '--color=no', '--tb=short'], cwd=tmp_path, capture_output=True, text=True, timeout=30,
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).parent), 'PYTEST_ADDOPTS': ''})
    output = re.sub(r'\x1b\[[0-9;?]*[a-zA-Z]', '', result.stdout)
    assert result.returncode == 1, result.stdout + result.stderr
    assert re.search(r'test_first.py\s+\[=+\] 3/3\s+F:1\s+s/x:1', output)
    assert re.search(r'test_second.py\s+\[=+\] 2/2\s+F:1\s+s/x:1', output)
    assert '1 failed, 1 passed, 1 skipped, 1 xfailed, 1 error' in output
    assert 'failure detail preserved' in output
    assert 'setup detail preserved' in output
