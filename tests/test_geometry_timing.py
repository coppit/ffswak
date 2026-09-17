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


@pytest.mark.parametrize('transition', [0, .25, .5])
@pytest.mark.parametrize('order', ['repeat', 'repeat-interior', 'backward', 'nested', 'separated-by-other-file'])
def test_requested_playback_order_survives_transition_adjustment(media, transition, order):
    # Compare every frame against independently assembled trims and blends. A ramp
    # identifies source time, so losing/reordering frames cannot hide in a solid scene.
    seconds = 6
    values = 40 + np.arange(seconds * FPS, dtype=np.uint8)
    frames = np.broadcast_to(values[:, None, None, None], (seconds * FPS, 120, 160, 3))
    source = media.encode('clock.mov', frames)
    other = media.encode('other.mov', frames)
    ranges = {
        'repeat': [(source, 0, 6), (source, 0, 6)],
        'repeat-interior': [(source, 1, 3), (source, 1, 3)],
        'backward': [(source, 3, 5), (source, 1, 4)],
        'nested': [(source, 1, 5), (source, 2, 4)],
        'separated-by-other-file': [(source, 1, 3), (other, 0, 2), (source, 2, 4)],
    }[order]
    args = []
    clips = []
    for index, (path, start, end) in enumerate(ranges):
        args += [path, f'{start}-{end}']
        extended_start = max(0, start - transition) if index else start
        extended_end = min(seconds, end + transition) if index < len(ranges)-1 else end
        clips.append(values[round(extended_start*FPS):round(extended_end*FPS)].astype(float))
    expected = clips[0]
    overlap_frames = round(transition * FPS)
    for clip in clips[1:]:
        if overlap_frames:
            fraction = np.arange(overlap_frames) / overlap_frames
            blend = expected[-overlap_frames:] * (1-fraction) + clip[:overlap_frames] * fraction
            expected = np.concatenate([expected[:-overlap_frames], blend, clip[overlap_frames:]])
        else:
            expected = np.concatenate([expected, clip])
    output = media.process('-T', str(transition), *args)
    assert_timeline(media, output, len(expected)/FPS, (160, 120))
    actual = media.decode(output)[:, 16:-16, 16:-16].mean(axis=(1, 2, 3))
    np.testing.assert_allclose(actual, expected, atol=4)


@pytest.mark.parametrize('transition', [0, .25, .5])
def test_overlapping_requested_ranges_keep_content_with_zero_and_nonzero_fades(media, transition):
    # Requested overlap is .5s. These transitions require no additional source
    # padding after correction; t=0 must concatenate both requested ranges in full.
    values = 40 + np.arange(6 * FPS, dtype=np.uint8)
    source = media.encode('clock.mov', np.broadcast_to(values[:, None, None, None], (6*FPS,120,160,3)))
    first, second = values[24:72].astype(float), values[60:96].astype(float)
    n = round(transition * FPS)
    if n:
        weight = np.arange(n) / n
        expected = np.concatenate([first[:-n], first[-n:]*(1-weight)+second[:n]*weight, second[n:]])
    else:
        expected = np.concatenate([first, second])
    output = media.process('-T', str(transition), source, '1-3', source, '2.5-4')
    assert_timeline(media, output, len(expected)/FPS, (160,120))
    actual = media.decode(output)[:,16:-16,16:-16].mean(axis=(1,2,3))
    np.testing.assert_allclose(actual, expected, atol=4)


@pytest.mark.parametrize('speed,expected_ranges', [(1, [(1,3),(2.5,4)]), (2, [(1,3.25),(2.25,4)])])
def test_transition_overlap_is_measured_in_source_seconds(media, speed, expected_ranges):
    # .5 output seconds means 1 source second at 2x. Subtracting the raw transition
    # duration from source times produces the wrong ranges and output length.
    values = 40 + np.arange(6 * FPS, dtype=np.uint8)
    source = media.encode('clock.mov', np.broadcast_to(values[:,None,None,None], (6*FPS,120,160,3)))
    first, second = [values[round(start*FPS):round(end*FPS):speed].astype(float)
                     for start,end in expected_ranges]
    weight = np.arange(12)/12
    expected = np.concatenate([first[:-12], first[-12:]*(1-weight)+second[:12]*weight, second[12:]])
    output = media.process('-p', str(speed), source, '1-3', source, '2.5-4')
    assert_timeline(media, output, len(expected)/FPS, (160,120))
    actual = media.decode(output)[:,16:-16,16:-16].mean(axis=(1,2,3))
    np.testing.assert_allclose(actual, expected, atol=4)


@pytest.mark.parametrize('transition', [0, .25, .5])
@pytest.mark.parametrize('speeds', [(1, 2), (2, 1)])
def test_different_speeds_meet_at_transition_midpoint(media, transition, speeds):
    # Adjacent requested ranges meet at source time 3. The fade should be
    # centered there, with each side contributing half a fade at its own speed.
    a, b = speeds
    ranges = [(1, 3 + transition*a/2, a), (3 - transition*b/2, 5, b)]
    assert_different_speed_transition(media, transition, speeds, ['1-3', '3-5'], ranges)


@pytest.mark.parametrize('speeds,requested,expected', [
    # Incoming padding is capped by the file start. Retract the remaining
    # correction from the outgoing side without trimming either requested range.
    ((1, 2), ['0-.75', '.25-4'], [(0, 1, 1), (.25, 4, 2)]),
    # Outgoing padding is capped by the file end: the converse case.
    ((2, 1), ['1-5.75', '5.25-6'], [(1, 5.75, 2), (5, 6, 1)]),
    # Requested overlap alone exceeds the desired overlap. Remove all padding,
    # but preserve the user's overlapping content in both speed orders.
    ((1, 2), ['1-3', '2-5'], [(1, 3, 1), (2, 5, 2)]),
    ((2, 1), ['1-3', '2-5'], [(1, 3, 2), (2, 5, 1)]),
])
def test_different_speeds_respect_available_transition_padding(media, speeds, requested, expected):
    assert_different_speed_transition(media, .5, speeds, requested, expected)


def assert_different_speed_transition(media, transition, speeds, requested, expected_ranges):
    values = 40 + np.arange(6 * FPS, dtype=np.uint8)
    source = media.encode('clock.mov', np.broadcast_to(values[:, None, None, None], (6*FPS, 120, 160, 3)))
    first, second = [values[round(start*FPS):round(end*FPS):speed].astype(float)
                     for start, end, speed in expected_ranges]
    n = round(transition * FPS)
    if n:
        weight = np.arange(n) / n
        expected = np.concatenate([first[:-n], first[-n:]*(1-weight)+second[:n]*weight, second[n:]])
    else:
        expected = np.concatenate([first, second])
    output = media.process('-T', str(transition), '--',
                           '-p', str(speeds[0]), source, requested[0],
                           '-p', str(speeds[1]), source, requested[1])
    assert_timeline(media, output, len(expected)/FPS, (160, 120))
    actual = media.decode(output)[:, 16:-16, 16:-16].mean(axis=(1, 2, 3))
    np.testing.assert_allclose(actual, expected, atol=4)
