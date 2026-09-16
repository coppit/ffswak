"""Targeted historical regressions for data that is awkward to encode on demand."""
import importlib.util
import signal
import sys
import threading
from types import SimpleNamespace

import pytest
from conftest import ROOT


@pytest.fixture
def app():
    # Import safely without running the CLI; restore the exception hooks installed by Rich.
    spec = importlib.util.spec_from_file_location('ffswak_under_test', ROOT / 'ffswak.py')
    module = importlib.util.module_from_spec(spec)
    old_sys, old_thread = sys.excepthook, threading.excepthook
    spec.loader.exec_module(module)
    yield module
    sys.excepthook = old_sys
    threading.excepthook = old_thread


@pytest.fixture
def probe_data():
    return {'format': {'duration': '2.0', 'bit_rate': '500000'}, 'streams': [
        {'codec_type': 'video', 'width': 320, 'height': 240, 'pix_fmt': 'yuv420p',
         'avg_frame_rate': '24000/1001'}]}


def clip(app):
    return app.Clip(input_file='synthetic.mov', start=0, end=None, speedup=1,
                    volume=1, stabilize=False, interlace_test=False)


def test_missing_stream_bitrate_uses_container_bitrate(app, monkeypatch, probe_data):
    # f6a93ca: an actual ffprobe-shaped record with no per-stream bit_rate.
    monkeypatch.setattr(app, 'ffprobe', lambda _: probe_data)
    result = clip(app)
    assert result.video_bitrate == 500000
    assert result.avg_frame_rate == app.Fraction(24000, 1001)


@pytest.mark.parametrize('field,value,diagnostic', [
    ('avg_frame_rate', '0/0', 'frame rate'),
    ('width', None, 'video width'),
    ('pix_fmt', None, 'video pixel format'),
    ('bit_rate', 'N/A', 'video bitrate'),
])
def test_invalid_probe_metadata_reports_field_instead_of_traceback(app, monkeypatch, probe_data, field, value, diagnostic):
    # d7688f8: malformed metadata should become a useful input error.
    probe_data['streams'][0][field] = value
    monkeypatch.setattr(app, 'ffprobe', lambda _: probe_data)
    messages = []
    monkeypatch.setattr(app, 'cprint', lambda message: messages.append(message))
    with pytest.raises(SystemExit) as error:
        clip(app)
    assert error.value.code == 1
    assert diagnostic in ' '.join(messages)
    assert 'synthetic.mov' in ' '.join(messages)


def test_audio_copy_decision_uses_its_own_video(app, monkeypatch):
    # cd34ef0 used the global video rather than self. Two different videos catch that.
    own = app.Video('/tmp', None, app.Dimensions(320, 240), 24)
    own.append(SimpleNamespace(audio_bitrate=128000, audio_filters=[]))
    other = app.Video('/tmp', None, app.Dimensions(320, 240), 24)
    other.append(SimpleNamespace(audio_bitrate=300000, audio_filters=[]))
    monkeypatch.setattr(app, 'video', other, raising=False)
    assert own.can_copy_audio
    assert not other.can_copy_audio
    own[0].audio_filters = [('volume', [0.5], {})]
    assert not own.can_copy_audio


def test_subprocess_output_drains_both_pipes_and_retains_partial_lines(app):
    # b89aac7: exceed the read buffer, emit invalid UTF-8, end without newline, and exit fast.
    code = "import os; os.write(1, b'x'*20000+b'\\ntail'); os.write(2, b'progress\\rdiagnostic\\xff')"
    original = signal.getsignal(signal.SIGINT)
    stdout, stderr = app.run_ffmpeg([sys.executable, '-c', code], None)
    assert stdout == ['x'*20000, 'tail']
    assert stderr == ['progress', 'diagnostic\ufffd']
    assert signal.getsignal(signal.SIGINT) == original


def test_failed_subprocess_reports_final_diagnostic_and_exit_code(app, monkeypatch):
    messages = []
    monkeypatch.setattr(app, 'cprint', lambda *args, **kwargs: messages.extend(map(str, args)))
    monkeypatch.setattr(app, 'eprint', lambda *args, **kwargs: messages.extend(map(str, args)))
    original = signal.getsignal(signal.SIGINT)
    code = "import os; os.write(2, b'actual encoder failure'); raise SystemExit(7)"
    with pytest.raises(SystemExit) as error:
        app.run_ffmpeg([sys.executable, '-c', code], None)
    assert error.value.code == 7
    assert 'actual encoder failure' in '\n'.join(messages)
    assert signal.getsignal(signal.SIGINT) == original


def test_progress_display_stops_before_failure_diagnostics(app, monkeypatch):
    # ef9ffe7: assert ordering, not terminal escape sequences or cosmetic formatting.
    events = []
    live = SimpleNamespace(transient=False, stop=lambda: events.append('stop'))
    progress = SimpleNamespace(live=live)
    monkeypatch.setattr(app, 'cprint', lambda *args, **kwargs: events.append('diagnostic'))
    monkeypatch.setattr(app, 'eprint', lambda *args, **kwargs: events.append('diagnostic'))
    with pytest.raises(SystemExit):
        app.run_ffmpeg([sys.executable, '-c', 'raise SystemExit(2)'], None, progress)
    assert live.transient
    assert events[0] == 'stop'
    assert 'diagnostic' in events[1:]
