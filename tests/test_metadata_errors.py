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


@pytest.mark.parametrize('value,limit_type,width,height', [
    ('1280x720', 'PIXELS', 1280, 720),
    ('.5', 'RELATIVE', .5, .5),
    ('.5,1', 'RELATIVE', .5, 1),
    ('16:9', 'ASPECT', 16, 9),
])
def test_dimension_limit_parser_normalizes_every_supported_syntax(app, value, limit_type, width, height):
    limit = app.dimensions_type(value)
    assert limit.type == app.DimensionLimitType[limit_type]
    assert (limit.width, limit.height) == (width, height)


@pytest.mark.parametrize('parser,value', [
    ('dimensions_type', '1280xwide'),
    ('crop_size_type', '1xwide'),
    ('crop_location_type', 'sideways'),
    ('time_range_type', 'start-end'),
    ('positive_int_type', 'zero'),
])
def test_option_parsers_report_invalid_values(app, parser, value):
    with pytest.raises(app.argparse.ArgumentTypeError):
        getattr(app, parser)(value)


def test_temporary_file_owner_removes_paths_it_created(app):
    temporary_files = app.TemporaryFiles()
    path = temporary_files.create('input.mov')
    assert not app.os.path.exists(path)
    with open(path, 'w') as file:
        file.write('temporary data')
    temporary_files.cleanup()
    assert not app.os.path.exists(path)


def test_compute_fps_uses_the_supplied_video_not_module_state(app, monkeypatch):
    clip = SimpleNamespace(avg_frame_rate=24)
    monkeypatch.setattr(app, 'video', [object(), object()], raising=False)
    assert app.compute_fps([clip], clip, 24) == []
    filters = app.compute_fps([clip, object()], clip, 24)
    assert filters == [('fps', [], {'fps': 24})]
    assert isinstance(filters[0], app.Filter)


def test_tripod_implies_stabilization_and_uses_each_filter_option_type(app):
    clip = SimpleNamespace(stabilize=False, tripod=1, video_bitrate=100000, mincontrast=.1, shakiness=8,
                           avg_frame_rate=app.Fraction(24, 1), transforms_file='transforms.trf', smoothing=20)
    app.Clip._set_stabilization_parameters(clip)
    assert clip.stabilize
    assert app.compute_stabilize(clip, 'prep', False) == [
        ('vidstabdetect', [], {'tripod': 24, 'mincontrast': .1, 'shakiness': 8, 'result': 'transforms.trf'})]
    assert app.compute_stabilize(clip, 'encode', False) == [
        ('vidstabtransform', [], {'input': 'transforms.trf', 'tripod': True})]


@pytest.mark.parametrize('seconds,precision,expected', [
    (3.3, 2, '0:03.3'),
    (59.999, 2, '1:00'),
    (0.5, 2, '0:00.5'),
])
def test_time_display_rounds_float_artifacts_and_carries(app, seconds, precision, expected):
    assert app.in_hms(seconds, precision) == expected


def test_rich_highlighter_does_not_split_timestamps(app):
    timestamp = app.Text('0:03.3')
    ipv6 = app.Text('2001:db8::1')
    number = app.Text('0.5')
    highlighter = app.ReprHighlighter()

    highlighter.highlight(timestamp)
    highlighter.highlight(ipv6)
    highlighter.highlight(number)

    assert timestamp.spans == []
    assert any(span.style == 'repr.ipv6' for span in ipv6.spans)
    assert any(span.style == 'repr.number' for span in number.spans)


def test_clip_snapshot_preserves_the_requested_reversed_range(app):
    clip = app.Clip.__new__(app.Clip)
    clip.__dict__.update(index=0, input_file='a.mov', start=3.3, end=10, speedup=1, reverse=True)
    snapshot = app.copy.copy(clip)
    clip.start = 2.9
    assert str(snapshot) == 'Clip 0 (a.mov 0:10-0:03.3)'


@pytest.mark.parametrize('output,expected', [
    ('Multi frame detection: TFF:   360 BFF:     0 Progressive:     0 Undetermined:     0', 'TFF'),
    ('Multi frame detection: TFF:     0 BFF:   360 Progressive:     0 Undetermined:     0', 'BFF'),
    ('Multi frame detection: TFF:     0 BFF:     0 Progressive:   360 Undetermined:     0', 'PROGRESSIVE'),
    ('Multi frame detection: TFF:     8 BFF:     8 Progressive:     0 Undetermined:   344', 'UNKNOWN'),
    ('Multi frame detection: TFF:   150 BFF:   150 Progressive:    60 Undetermined:     0', 'UNKNOWN'),
])
def test_idet_classification_uses_both_parities_and_rejects_ambiguity(app, output, expected):
    interlace_type, counts = app.classify_idet_output(output)
    assert interlace_type == app.InterlaceType[expected]
    assert sum(counts.values()) == 360


@pytest.mark.parametrize('interlace_type,expected', [
    ('TFF', [('yadif', [], {'parity': 'tff'})]),
    ('BFF', [('yadif', [], {'parity': 'bff'})]),
    ('PROGRESSIVE', []),
    ('TELECINE', []),
    ('UNKNOWN', []),
])
def test_deinterlacing_uses_detected_field_order(app, interlace_type, expected):
    clip = SimpleNamespace(interlace_type=app.InterlaceType[interlace_type])
    assert app.compute_deinterlace(clip) == expected


@pytest.mark.integration
@pytest.mark.parametrize('filename,expected', [
    ('bt601-525_480_interlaced_tff.mkv', 'TFF'),
    ('bt601-525_480_interlaced_bff.mkv', 'BFF'),
    ('bt601-525_480_progressive.mkv', 'PROGRESSIVE'),
    ('bt601-525_480_telecined_hard.mkv', 'TELECINE'),
    ('bt601-525_480_telecined_soft.mkv', 'PROGRESSIVE'),
])
def test_upstream_interlace_patterns(app, monkeypatch, filename, expected):
    fixture = ROOT / 'tests' / 'fixtures' / 'interlacing' / filename
    monkeypatch.setattr(app.time, 'sleep', lambda _: None)
    app.detect_interlace.cache_clear()
    assert app.detect_interlace(True, str(fixture)) == app.InterlaceType[expected]


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
