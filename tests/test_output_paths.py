"""Output naming must protect existing files through encoding and copy fallback."""
import importlib.util
import re
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ROOT, run
from test_video import quadrants


@pytest.mark.integration
@pytest.mark.parametrize('collision', [False, True])
def test_explicit_output_path_is_preserved_or_uniquified(media, collision):
    # One small frame is enough to exercise real output creation and collision protection.
    source = media.encode('source.mov', np.zeros((1, 64, 64, 3), dtype=np.uint8))
    base = media.directory / 'outputs'
    requested = base / 'nested' / 'chosen.mp4'
    requested.parent.mkdir(parents=True)
    sentinel = b'pre-existing output must survive'
    if collision:
        requested.write_bytes(sentinel)
    # Crop forces encoding, independently of the copy fallback tests below.
    run([sys.executable, ROOT / 'ffswak.py', '-cs', '.5', '-O', base, '-o', 'nested/chosen.mp4', source],
        cwd=media.directory)
    if collision:
        assert requested.read_bytes() == sentinel
        outputs = list(requested.parent.glob('chosen-*.mp4'))
        assert len(outputs) == 1
        assert re.fullmatch(r'chosen-[0-9a-f]{3}\.mp4', outputs[0].name)
        output = outputs[0]
    else:
        output = requested
        assert not list(requested.parent.glob('chosen-*.mp4'))
    assert output.is_file()
    assert output.stat().st_size > 0


@pytest.fixture
def app():
    spec = importlib.util.spec_from_file_location('ffswak_paths', ROOT / 'ffswak.py')
    module = importlib.util.module_from_spec(spec)
    old_sys, old_thread = sys.excepthook, threading.excepthook
    spec.loader.exec_module(module)
    yield module
    sys.excepthook = old_sys
    threading.excepthook = old_thread


@pytest.mark.parametrize('explicit', [True, False])
def test_name_collision_retries_and_resolution_stays_fixed(app, tmp_path, monkeypatch, explicit):
    from types import SimpleNamespace
    requested = tmp_path / 'movie.mp4'
    requested.write_bytes(b'existing')
    (tmp_path / 'movie-123.mp4').write_bytes(b'also existing')
    draws = iter([0x123, 0xabc])
    monkeypatch.setattr(app.random, 'randrange', lambda _: next(draws))
    video = app.Video(str(tmp_path), str(requested) if explicit else None, app.Dimensions(320, 240), 24)
    video.append(SimpleNamespace(input_file='movie.mov', video_bitrate=500000, audio_bitrate=None))
    video.clips_adjusted = True
    resolved = tmp_path / 'movie-abc.mp4'
    assert video.output_file == str(resolved)
    resolved.write_bytes(b'encoded output')
    assert video.output_file == str(resolved)
    assert requested.read_bytes() == b'existing'
    assert (tmp_path / 'movie-123.mp4').read_bytes() == b'also existing'


@pytest.mark.integration
@pytest.mark.parametrize('extension', ['mov', 'mp4', 'm4v'])
def test_copy_fallback_keeps_resolved_destination_and_container(media, extension):
    # Force the fallback decision to avoid tying this regression to encoder sizes.
    # All probing, encoding, copying/remuxing, and decoding still use real media.
    source = media.mux_audio(media.encode('source.mov', quadrants()))
    destinations = media.directory / 'outputs'
    destinations.mkdir()
    requested = destinations / f'chosen.{extension}'
    requested.write_bytes(b'existing requested file')
    basename_collision = destinations / source.name
    basename_collision.write_bytes(b'unrelated file with input basename')
    driver = (
        'import sys; sys.path.insert(0, sys.argv.pop(1)); import ffswak; '
        'ffswak.encoded_file_not_much_smaller = lambda video: True; ffswak.main()'
    )
    run([sys.executable, '-c', driver, ROOT, '-o', requested, source], cwd=media.directory)
    assert requested.read_bytes() == b'existing requested file'
    assert basename_collision.read_bytes() == b'unrelated file with input basename'
    outputs = list(destinations.glob(f'chosen-*.{extension}'))
    assert len(outputs) == 1
    output = outputs[0]
    assert media.decode(output).shape == (48, 240, 320, 3)
    if extension == 'mov':
        assert output.read_bytes() == source.read_bytes()
    else:
        # MOV -> MP4 must remux, not put a MOV file under an MP4 filename.
        assert output.read_bytes()[8:12] != b'qt  '
        assert media.probe(output)['streams'][0]['codec_name'] == 'h264'


@pytest.mark.parametrize('case', ['relative', 'relative-with-dir', 'absolute', 'absolute-with-dir', 'dir-only'])
@pytest.mark.parametrize('collision', [False, True])
def test_output_directory_and_filename_resolution(app, tmp_path, monkeypatch, case, collision):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / 'first-dir'
    absolute = tmp_path / 'absolute-dir' / 'file.mp4'
    if case == 'relative':
        output_dir, requested, expected = None, 'nested/file.mp4', tmp_path / 'nested/file.mp4'
    elif case == 'relative-with-dir':
        output_dir, requested, expected = str(base), 'nested/file.mp4', base / 'nested/file.mp4'
    elif case == 'absolute':
        output_dir, requested, expected = None, str(absolute), absolute
    elif case == 'absolute-with-dir':
        output_dir, requested, expected = str(base), str(absolute), absolute
    else:
        output_dir, requested, expected = str(base), None, base / 'source.mp4'
    expected.parent.mkdir(parents=True, exist_ok=True)
    if collision:
        expected.write_bytes(b'keep existing output')
    video = app.Video(output_dir, requested, app.Dimensions(320, 240), 24)
    video.append(SimpleNamespace(input_file='source.mov', video_bitrate=500000, audio_bitrate=None))
    video.clips_adjusted = True
    output = Path(video.output_file)
    if collision:
        assert expected.read_bytes() == b'keep existing output'
        assert output.parent == expected.parent
        assert re.fullmatch(re.escape(expected.stem) + r'-[0-9a-f]{3}' + re.escape(expected.suffix), output.name)
    else:
        assert output == expected
    assert not output.exists()


def test_default_output_directory_only_applies_to_generated_names(app, tmp_path, monkeypatch):
    from types import SimpleNamespace
    default = tmp_path / 'default'
    monkeypatch.setattr(app, 'DEFAULT_OUTPUT_DIR', str(default))
    monkeypatch.chdir(tmp_path)
    for requested, expected in [(None, default / 'movie.mp4'), ('nested/file.mp4', tmp_path / 'nested/file.mp4')]:
        video = app.Video(None, requested, app.Dimensions(320, 240), 24)
        video.append(SimpleNamespace(input_file='movie.mov', video_bitrate=500000, audio_bitrate=None))
        video.clips_adjusted = True
        assert video.output_file == str(expected)


@pytest.mark.integration
@pytest.mark.parametrize('extension', ['m4v', 'M4V'])
def test_m4v_output_supports_hevc(media, extension):
    # The extension must not select the HEVC-incompatible iPod muxer.
    from test_video import assert_color, COLORS
    source = media.encode('source.mov', quadrants())
    output = media.process('-cs', '.5', '-cl', '0,0', source, name=f'output.{extension}')
    stream = media.probe(output)['streams'][0]
    assert stream['codec_name'] == 'hevc'
    assert stream['codec_tag_string'] == 'hvc1'
    frames = media.decode(output)
    assert frames.shape == (48, 120, 160, 3)
    assert_color(frames[:, 8:-8, 8:-8], COLORS[0])
