"""Behavioral tests against the public CLI, with independent decoded-pixel oracles."""
import json

import numpy as np
import pytest

from conftest import FPS

pytestmark = pytest.mark.integration
COLORS = np.array([[230, 30, 30], [30, 220, 30], [30, 30, 230], [220, 220, 30]], dtype=np.uint8)


def quadrants(seconds=2):
    frame = np.empty((240, 320, 3), dtype=np.uint8)
    frame[:120, :160], frame[:120, 160:] = COLORS[:2]
    frame[120:, :160], frame[120:, 160:] = COLORS[2:]
    return np.repeat(frame[None], seconds * FPS, axis=0)


def assert_color(pixels, expected, tolerance=18):
    # Check every frame and 99% of each patch, not just an average that can hide corruption.
    errors = np.abs(pixels.astype(float) - expected)
    per_frame = np.quantile(errors, .99, axis=(1, 2, 3))
    assert np.max(per_frame) <= tolerance, f'Worst frame RGB error: {np.max(per_frame):.1f}'


def assert_timeline(media, output, seconds, dimensions):
    info = media.probe(output)
    stream = info['streams'][0]
    assert (stream['width'], stream['height']) == dimensions
    assert stream['codec_name'] == 'hevc'
    timestamps = np.array([float(frame['best_effort_timestamp_time']) for frame in info['frames']])
    assert len(timestamps) == round(seconds * FPS)
    np.testing.assert_allclose(timestamps, np.arange(len(timestamps)) / FPS, atol=1e-4)
    assert float(stream['duration']) == pytest.approx(seconds, abs=1 / FPS)


def test_per_input_crop_flags(media):
    # Isolate cuts with -T 0: top-left red followed by bottom-right yellow.
    first = media.encode('input1.mov', quadrants())
    second = media.encode('input2.mov', quadrants())
    output = media.process('-T', '0', '--', '-cs', '.5', '-cl', '0,0', first,
                           '-cs', '.5', '-cl', '1,1', second)
    assert_timeline(media, output, 4, (160, 120))
    frames = media.decode(output)
    assert_color(frames[:2 * FPS, 8:-8, 8:-8], COLORS[0])
    assert_color(frames[2 * FPS:, 8:-8, 8:-8], COLORS[3])


@pytest.mark.parametrize('repeat_filename', [True, False], ids=['repeated-file', 'multiple-ranges'])
def test_disjoint_ranges_from_same_input(media, repeat_filename):
    # Each second is identifiable, including both seconds that must be omitted.
    palette = np.vstack([COLORS, [220, 30, 220], [30, 220, 220]]).astype(np.uint8)
    frames = np.repeat(palette[:, None, None, :], FPS, axis=0)
    frames = np.broadcast_to(frames, (6 * FPS, 240, 320, 3))
    source = media.encode('timeline.mov', frames)
    args = [source, '0-2', source, '4-5'] if repeat_filename else [source, '0-2', '4-5']
    output = media.process('-T', '0', *args)
    assert_timeline(media, output, 3, (320, 240))
    actual = media.decode(output)
    for index, color in enumerate(palette[[0, 1, 4]]):
        assert_color(actual[index * FPS:(index + 1) * FPS, 16:-16, 16:-16], color)


def test_mixed_bt601_h264_8bit_and_bt709_prores_10bit(media):
    # Exercise distinct SDR color matrices, bit depths and chroma sampling, not HDR
    # or preservation of low-order 10-bit gradient information.
    low = media.encode('bt601-8bit.mov', quadrants())
    yuv = media.encode('bt709-10bit.mov', quadrants(), encoding='prores10')
    low_info = media.probe(low)['streams'][0]
    assert low_info['codec_name'] == 'h264'
    assert low_info['pix_fmt'] == 'yuv420p'
    assert low_info['color_space'] == 'smpte170m'
    high = media.probe(yuv)['streams'][0]
    assert high['codec_name'] == 'prores'
    assert high['pix_fmt'] == 'yuv422p10le'
    assert high['color_space'] == 'bt709'
    output = media.process('-T', '0', low, yuv)
    assert_timeline(media, output, 4, (320, 240))
    assert media.probe(output)['streams'][0]['pix_fmt'] == 'yuv422p10le'
    frames = media.decode(output)
    for color, (y, x) in zip(COLORS, [(60, 80), (60, 240), (180, 80), (180, 240)]):
        assert_color(frames[:, y-20:y+20, x-20:x+20], color)


def landmark_centers(frames):
    red = frames[..., 0].astype(float)
    mask = (red > 160) & (red > frames[..., 1] * 1.8) & (red > frames[..., 2] * 1.8)
    positions = []
    for frame in mask:
        y, x = np.nonzero(frame)
        assert len(x) > 100, 'Landmark lost; blank/overcropped output is not stabilization'
        positions.append([x.mean(), y.mean()])
    return np.array(positions)


def jitter(positions):
    # Second differences isolate rapid shake from slow camera drift.
    return np.sqrt(np.mean(np.diff(positions, n=2, axis=0) ** 2))


def test_stabilization_reduces_camera_shake(media, stabilization_available):
    # Use a full eight-second clip so smoothing has ample context.
    seconds = 8
    rng = np.random.default_rng(831)
    texture = rng.integers(35, 150, (35, 45), dtype=np.uint8)
    texture = np.repeat(np.repeat(texture, 8, axis=0), 8, axis=1)
    scene = np.repeat(texture[..., None], 3, axis=2)
    scene[125:155, 165:195] = [240, 20, 20]
    offsets = rng.integers(-7, 8, (seconds * FPS, 2))
    frames = np.stack([scene[20+y:260+y, 20+x:340+x] for y, x in offsets])
    source = media.encode('shaky.mov', frames)
    output = media.process('-s', source)
    assert_timeline(media, output, seconds, (320, 240))
    # Exclude smoothing startup/end effects from the motion measurement.
    before = landmark_centers(media.decode(source))[FPS:-FPS]
    decoded = media.decode(output)
    after = landmark_centers(decoded)[FPS:-FPS]
    initial, stabilized = jitter(before), jitter(after)
    (media.directory / 'motion.json').write_text(
        json.dumps({'input_jitter_px': initial, 'output_jitter_px': stabilized}, indent=2))
    assert initial > 5, f'Fixture is not sufficiently shaky: {initial:.2f}'
    assert stabilized < initial * .35, f'Jitter did not improve enough: {initial:.2f} -> {stabilized:.2f} px'
    assert stabilized < 2, f'Residual jitter is too high: {stabilized:.2f} px'
    assert np.std(decoded[..., 1]) > 15, 'Scene detail lost'


def test_default_crossfade_blends_frames(media):
    red = np.broadcast_to(COLORS[0], (2 * FPS, 240, 320, 3))
    blue = np.broadcast_to(COLORS[2], (2 * FPS, 240, 320, 3))
    first = media.encode('red.mov', red)
    second = media.encode('blue.mov', blue)
    output = media.process(first, second)
    assert_timeline(media, output, 3.5, (320, 240))
    frames = media.decode(output)
    assert_color(frames[:36, 16:-16, 16:-16], COLORS[0])
    assert_color(frames[48:, 16:-16, 16:-16], COLORS[2])
    # The half-second overlap starts at 1.5 seconds; check each blend independently.
    for index in range(1, 12):
        fraction = index / 12
        expected = COLORS[0] * (1 - fraction) + COLORS[2] * fraction
        assert_color(frames[36+index:37+index, 16:-16, 16:-16], expected)
