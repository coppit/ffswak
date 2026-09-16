"""Output oracles for the heavily instrumented sizing and transition calculations."""
import numpy as np
import pytest
from conftest import FPS, run
from test_video import COLORS, quadrants, assert_color, assert_timeline

pytestmark = pytest.mark.integration


def test_portrait_landscape_join_preserves_proportions_and_padding(media):
    landscape = media.encode('landscape.mov', quadrants())
    portrait = media.encode('portrait.mov', np.rot90(quadrants(), k=1, axes=(1, 2)))
    output = media.process('-T', '0', '-D', '240x240', landscape, portrait)
    assert_timeline(media, output, 4, (240, 240))
    frames = media.decode(output)
    # 4:3 and 3:4 become 240x180 and 180x240, centered with 30-pixel bars.
    assert_color(frames[:48, :25, :], np.zeros(3))
    assert_color(frames[:48, -25:, :], np.zeros(3))
    assert_color(frames[48:, :, :25], np.zeros(3))
    assert_color(frames[48:, :, -25:], np.zeros(3))
    for y, x, color in [(60, 40, 0), (60, 200, 1), (180, 40, 2), (180, 200, 3)]:
        assert_color(frames[:48, y-8:y+8, x-8:x+8], COLORS[color])
    for y, x, color in [(40, 60, 1), (40, 180, 3), (200, 60, 0), (200, 180, 2)]:
        assert_color(frames[48:, y-8:y+8, x-8:x+8], COLORS[color])


def test_phone_rotation_metadata_is_applied_once(media):
    source = media.encode('landscape.mov', quadrants())
    rotated = media.directory / 'phone.mov'
    run(['ffmpeg', '-v', 'error', '-display_rotation', '90', '-i', source,
         '-c', 'copy', rotated], cwd=media.directory)
    assert any(s.get('rotation') == 90 for s in media.probe(rotated)['streams'][0]['side_data_list'])
    output = media.process('-D', '160x120', rotated)
    assert_timeline(media, output, 2, (120, 160))
    frames = media.decode(output)
    for y, x, color in [(40, 30, 1), (40, 90, 3), (120, 30, 0), (120, 90, 2)]:
        assert_color(frames[:, y-8:y+8, x-8:x+8], COLORS[color])


@pytest.mark.parametrize('ranges,seconds,checkpoints', [
    (['1-2', '4-5'], 2.5, [(0, 1), (18, 1), (42, 0), (54, 0)]),
    (['5-6', '0-1'], 1.5, [(0, 1), (6, 1), (24, 0), (30, 0)]),
])
def test_transition_range_extensions_and_file_boundaries(media, ranges, seconds, checkpoints):
    # Each source second has a known color. Interior ranges can extend by .5s;
    # boundary ranges cannot. Validate duration and source content outside overlaps.
    palette = np.tile(COLORS, (2, 1))[:6]
    frames = np.broadcast_to(np.repeat(palette, FPS, axis=0)[:, None, None], (6*FPS, 240, 320, 3))
    source = media.encode('clock.mov', frames)
    output = media.process(source, ranges[0], source, ranges[1])
    assert_timeline(media, output, seconds, (320, 240))
    actual = media.decode(output)
    for frame, color in checkpoints:
        assert_color(actual[frame:frame+1, 16:-16, 16:-16], COLORS[color])


def test_overlapping_ranges_do_not_repeat_source_motion(media):
    # 1-3 and 2.5-4 overlap. The adjustment should yield one continuous 1-4 span.
    frames = np.zeros((6*FPS, 240, 320, 3), dtype=np.uint8) + 40
    for i, frame in enumerate(frames):
        x = 30 + i
        frame[100:140, x:x+20] = [230, 30, 30]
    source = media.encode('moving.mov', frames)
    output = media.process(source, '1-3', source, '2.5-4')
    assert_timeline(media, output, 3, (320, 240))
    decoded = media.decode(output)
    centers = []
    for frame in decoded:
        y, x = np.nonzero((frame[..., 0] > 150) & (frame[..., 1] < 70))
        assert len(x) > 500
        centers.append(x.mean())
    np.testing.assert_allclose(centers, 63.5 + np.arange(72), atol=1)


def test_odd_dimension_limit_rounds_to_encodable_size(media):
    source = media.encode('landscape.mov', quadrants())
    output = media.process('-D', '161x121', source)
    assert_timeline(media, output, 2, (160, 120))
    frames = media.decode(output)
    for y, x, color in [(30, 40, 0), (30, 120, 1), (90, 40, 2), (90, 120, 3)]:
        assert_color(frames[:, y-8:y+8, x-8:x+8], COLORS[color])
