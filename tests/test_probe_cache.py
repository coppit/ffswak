"""Probe reuse must not hide file changes or mix different metadata requests."""
import os

import pytest

import conftest


@pytest.fixture
def probes(tmp_path, monkeypatch):
    source = tmp_path / 'input.mov'
    source.write_bytes(b'video')
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        return b'{"streams": [{"codec_name": "h264"}], "frames": []}'

    monkeypatch.setattr(conftest, 'run', probe)
    return conftest.Media(tmp_path), source, calls


def test_probe_reuses_results_without_sharing_mutable_data(probes):
    media, source, calls = probes
    first = media.probe(source)
    first['streams'][0]['codec_name'] = 'changed'
    assert media.probe(source.name)['streams'][0]['codec_name'] == 'h264'
    assert len(calls) == 1


def test_probe_options_and_media_instances_have_separate_caches(probes):
    media, source, calls = probes
    media.probe(source)
    media.streams(source)
    media.streams(source)
    assert len(calls) == 2
    assert '-show_frames' in calls[0] and '-show_frames' not in calls[1]
    conftest.Media(media.directory).probe(source)
    assert len(calls) == 3


@pytest.mark.parametrize('change', ['resize', 'rewrite', 'timestamp', 'replace'])
def test_probe_invalidates_changed_files(probes, change):
    media, source, calls = probes
    media.probe(source)
    before = source.stat()
    if change == 'resize':
        source.write_bytes(b'longer video')
    elif change == 'rewrite':
        source.write_bytes(b'other')
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    elif change == 'timestamp':
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    else:
        replacement = source.with_suffix('.replacement')
        replacement.write_bytes(b'other')
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(source)
    media.probe(source)
    assert len(calls) == 2
