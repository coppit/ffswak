"""Regressions for stream selection, copy eligibility and missing-audio timelines."""
import subprocess

import numpy as np
import pytest
from conftest import FPS
from test_video import quadrants, assert_timeline, assert_color, COLORS

pytestmark = pytest.mark.integration


def segment(audio, start, end):
    return audio[round(start * 48000):round(end * 48000)]


def rms(audio):
    assert len(audio)
    return np.sqrt(np.mean(audio ** 2))


def assert_tone(audio, start, end, frequency):
    samples = segment(audio, start, end)
    assert rms(samples) > .03
    spectrum = abs(np.fft.rfft(samples * np.hanning(len(samples))))
    peak = np.argmax(spectrum) * 48000 / len(samples)
    assert peak == pytest.approx(frequency, abs=4)


@pytest.mark.parametrize('unsupported_first', [False, True], ids=['aac-before-apac', 'apac-before-aac'])
def test_selects_aac_while_ignoring_unsupported_apac_track(media, unsupported_first):
    # b5b19db: model the observed iPhone stream layout using only synthetic data.
    source = media.mux_audio(media.encode('scene.mov', quadrants()),
                             unsupported_first=unsupported_first)
    tracks = [s for s in media.streams(source) if s['codec_type'] == 'audio']
    assert len(tracks) == 2
    unknown = tracks[0 if unsupported_first else 1]
    assert unknown['codec_tag_string'] == 'apac'
    assert not unknown.get('codec_name')
    # Negative control: this fixture really rejects decoding all audio streams.
    all_audio = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(source),
                                '-map', '0:a', '-f', 'null', '-'],
                               capture_output=True, timeout=30)
    assert all_audio.returncode != 0
    assert b'decoder' in all_audio.stderr.lower()
    output = media.process('-cs', '.5', '-cl', '0,0', '-v', '.5', source)
    assert_timeline(media, output, 2, (160, 120))
    assert_color(media.decode(output)[:, 8:-8, 8:-8], COLORS[0])
    assert len([s for s in media.streams(output) if s['codec_type'] == 'audio']) == 1
    # Half volume is deliberately applied to force decoding rather than stream copy.
    samples = media.audio(output)
    assert_tone(samples, .2, 1.8, 440)
    assert rms(segment(samples, .2, 1.8)) == pytest.approx(.044, abs=.007)


def test_multiple_recognized_audio_tracks_selects_first_tone(media):
    source = media.mux_audio(media.encode('scene.mov', quadrants()), (440, 880))
    output = media.process('-cs', '.5', source)
    assert len([s for s in media.streams(output) if s['codec_type'] == 'audio']) == 1
    assert_tone(media.audio(output), .2, 1.8, 440)


@pytest.mark.parametrize('audible_first', [True, False])
def test_join_with_missing_audio_preserves_silence_and_sync(media, audible_first):
    silent = media.encode('silent.mov', quadrants())
    audible = media.mux_audio(media.encode('audible.mov', quadrants()), (660,))
    clips = [audible, silent] if audible_first else [silent, audible]
    output = media.process('-T', '0', *clips)
    assert_timeline(media, output, 4, (320, 240))
    samples = media.audio(output)
    assert len(samples) / 48000 == pytest.approx(4, abs=.05)
    tone_start, silence_start = (0, 2) if audible_first else (2, 0)
    assert_tone(samples, tone_start + .2, tone_start + 1.8, 660)
    assert rms(segment(samples, silence_start + .2, silence_start + 1.8)) < .002


@pytest.mark.parametrize('volume', [1, .5])
def test_audio_copy_and_volume_filter_preserve_tone(media, volume):
    # cd34ef0: no audio filter permits copy; volume changes must actually be applied.
    source = media.mux_audio(media.encode('scene.mov', quadrants()))
    output = media.process('-cs', '.5', '-v', str(volume), source)
    original, actual = media.audio(source), media.audio(output)
    assert_tone(actual, .2, 1.8, 440)
    ratio = rms(segment(actual, .2, 1.8)) / rms(segment(original, .2, 1.8))
    assert ratio == pytest.approx(volume, abs=.025)
    if volume == 1:
        np.testing.assert_array_equal(actual, original)


def test_trim_audio_matches_selected_video_seconds(media):
    # Cut a two-tone sequence and check the audio change occurs with the color change.
    from conftest import run
    first = media.mux_audio(media.encode('red.mov', np.broadcast_to(COLORS[0], (2*FPS,240,320,3))), (440,))
    second = media.mux_audio(media.encode('blue.mov', np.broadcast_to(COLORS[2], (2*FPS,240,320,3))), (880,))
    source = media.directory / 'sequence.mov'
    run(['ffmpeg', '-v', 'error', '-i', first, '-i', second, '-filter_complex',
         '[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]', '-map', '[v]', '-map', '[a]',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', source], cwd=media.directory)
    output = media.process(source, '1-3')
    assert_timeline(media, output, 2, (320,240))
    audio = media.audio(output)
    assert_tone(audio, .2, .8, 440)
    assert_tone(audio, 1.2, 1.8, 880)
    frames = media.decode(output)
    assert_color(frames[:FPS, 16:-16,16:-16], COLORS[0])
    assert_color(frames[FPS:, 16:-16,16:-16], COLORS[2])
