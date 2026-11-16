"""Source-byte accounting uses the encode's packets, with ffprobe only as a test oracle."""
import json
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import run
from test_metadata_errors import app  # noqa: F401


def clip(path, start, end, *, audio=None, video=True):
    return SimpleNamespace(input_file=str(path), start=start, end=end,
        input_dims=(160, 96) if video else None, video_filters=[], audio_filters=[],
        audio_stream_index=audio)


def oracle(path, start, end, streams):
    data = json.loads(run(['ffprobe', '-v', 'error', '-show_packets', '-show_streams', '-of', 'json', path],
        cwd=path.parent))
    timebases = {s['index']: Fraction(s['time_base']) for s in data['streams']}
    total = Fraction(0)
    for packet in data['packets']:
        if packet['stream_index'] not in streams:
            continue
        timebase = timebases[packet['stream_index']]
        pts = int(packet['pts']) * timebase
        duration = int(packet.get('duration', 0)) * timebase
        if duration > 0:
            overlap = max(0, min(Fraction(str(end)), pts + duration) - max(Fraction(str(start)), pts))
            total += int(packet['size']) * overlap / duration
        elif Fraction(str(start)) <= pts < Fraction(str(end)):
            total += int(packet['size'])
    return round(total)


@pytest.mark.integration
def test_vbr_packet_sizes_match_probe_with_concurrent_repeated_inputs(app, media):
    # A textured scene drifts slowly for two seconds, then moves rapidly. Encode
    # with CRF and B-frames so both variable bitrate and reordered PTS are exercised.
    rng = np.random.default_rng(4)
    texture = rng.integers(0, 256, (96, 160, 3), dtype=np.uint8)
    frames = np.array([np.roll(texture, i // 24 if i < 48 else 2 + (i - 48) * 9, axis=1)
                       for i in range(96)])
    source = media.mux_audio(media.encode('slow-fast.mp4', frames), frequencies=(440, 880))
    second = media.encode('second.mp4', frames[::-1])
    clips = [clip(source, 0.013, 1.987, audio=1), clip(source, 2.013, 3.987, audio=0),
             clip(second, 0.513, 3.487)]
    # Primary outputs really decode/encode while the accounting outputs copy the
    # same demuxed packets. There are only two -i options for three clips.
    command = ['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-i', str(second),
               '-map', '0:v:0', '-map', '1:v:0', '-c:v', 'libx264', '-f', 'null', '-']
    tally = app.SourcePacketTally(clips, command)
    try:
        stdout, stderr = app.run_ffmpeg(tally.command, None, packet_tally=tally)
        tally.finish()
        assert not stdout and not stderr  # Packet summaries never enter the retained log.
        assert tally.command.count('-i') == 2
        for item, streams in zip(clips, ({0, 2}, {0, 1}, {0})):
            assert item.source_packet_size == oracle(
                source if item is not clips[2] else second, item.start, item.end, streams)
        assert clips[1].source_packet_size > clips[0].source_packet_size * 2
    finally:
        tally.close()
    assert not tally.readers and not tally.writers


def test_packet_boundary_policy_and_missing_timestamps(app):
    item = clip('unused.mp4', 1, 2)
    tally = app.SourcePacketTally([item], ['ffmpeg', '-i', 'unused.mp4', '-f', 'null', '-'])
    try:
        tally.consume(0, '#tb 0: 1/1000')
        for pts, size in ((999, 10), (1000, 20), (1999, 30), (2000, 40)):
            tally.consume(0, f'0, 0, {pts}, 1, {size}, 0x1234')
        tally.finish()
        assert item.source_packet_size == 50
        tally.consume(0, '0, 0, NOPTS, 1, 50, 0x1234')
        tally.finish()
        assert item.source_packet_size is None
    finally:
        tally.close()


@pytest.mark.parametrize('packet_size, expected_copy, label', [
    (200, False, 'source media size, excluding container overhead'),
    (None, True, 'estimated original size'),
])
def test_size_comparison_uses_packet_total_or_estimate(app, tmp_path, monkeypatch,
                                                      packet_size, expected_copy, label):
    class Video(list):
        estimated_file_size = 50
        output_file = str(tmp_path / 'output.mp4')
    video = Video([SimpleNamespace(start=1, end=2, input_duration=3, source_packet_size=packet_size)])
    (tmp_path / 'output.mp4').write_bytes(bytes(100))
    messages = []
    monkeypatch.setattr(app, '_debug', False)
    monkeypatch.setattr(app, 'cprint', lambda text, **kwargs: messages.append(text))
    assert app.encoded_file_not_much_smaller(video) == expected_copy
    assert label in messages[0]


@pytest.mark.integration
def test_nonzero_container_start_uses_normalized_clip_timestamps(app, tmp_path):
    source = tmp_path / 'offset.mp4'
    run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=160x96:r=24:d=2',
         '-c:v', 'libx264', '-output_ts_offset', '7', source], cwd=tmp_path)
    item = clip(source, 0.25, 1.75)
    command = ['ffmpeg', '-v', 'error', '-i', str(source), '-map', '0:v:0', '-f', 'null', '-']
    tally = app.SourcePacketTally([item], command)
    try:
        app.run_ffmpeg(tally.command, None, packet_tally=tally)
        tally.finish()
        assert item.source_packet_size == oracle(source, 7.25, 8.75, {0})
    finally:
        tally.close()


def test_repeated_source_reuses_each_clips_existing_demuxers(app):
    first, second = clip('same.mp4', 0, 1, audio=0), clip('same.mp4', 2, 4, audio=0)
    first.encode_video_input, first.encode_audio_input = 'video-first', 'audio-first'
    second.encode_video_input, second.encode_audio_input = 'video-second', 'audio-second'
    command = ['ffmpeg', '-i', 'video-first', '-i', 'video-second',
               '-i', 'audio-first', '-i', 'audio-second', '-f', 'null', '-']
    tally = app.SourcePacketTally([first, second], command)
    try:
        added = tally.command[len(command):]
        assert [added[i + 1] for i, arg in enumerate(added[:-1]) if arg == '-map'] == [
            '0:v:0', '2:a:0', '1:v:0', '3:a:0']
    finally:
        tally.close()


@pytest.mark.parametrize('pts, duration, size, expected', [
    (500, 1000, 100, 50),   # Start cuts halfway through a packet.
    (1500, 1000, 100, 50),  # End cuts halfway through a packet.
    (500, 2000, 100, 50),   # Both cuts lie within one packet.
    (1000, 1000, 100, 100), # Exactly aligned boundaries.
    (0, 1000, 100, 0),     # Ends exactly at clip start.
    (2000, 1000, 100, 0),  # Starts exactly at clip end.
    (1100, 100, 37, 37),   # Interior packets count exactly.
    (1500, 0, 37, 37),     # Missing duration falls back to PTS membership.
    (500, 0, 37, 0),
    (1500, -1, 37, 37),
])
def test_partial_packet_overlap(app, pts, duration, size, expected):
    item = clip('unused.mp4', 1, 2)
    tally = app.SourcePacketTally([item], ['ffmpeg', '-i', 'unused.mp4', '-f', 'null', '-'])
    try:
        tally.consume(0, '#tb 0: 1/1000')
        tally.consume(0, f'0, 0, {pts}, {duration}, {size}, 0x1234')
        tally.finish()
        assert item.source_packet_size == expected
    finally:
        tally.close()


def test_fractional_bytes_are_rounded_after_accumulation(app):
    item = clip('unused.mp4', 1, 2)
    tally = app.SourcePacketTally([item], ['ffmpeg', '-i', 'unused.mp4', '-f', 'null', '-'])
    try:
        tally.consume(0, '#tb 0: 1/1000')
        tally.consume(0, '0, 0, 700, 1000, 1, 0x1234')
        tally.consume(0, '0, 0, 1300, 1000, 1, 0x1234')
        tally.finish()
        assert tally.sizes[0] == Fraction(7, 5)
        assert item.source_packet_size == 1
    finally:
        tally.close()
