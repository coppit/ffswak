"""Real media helpers; all generated files and command logs live in pytest's temp dir."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytest_plugins = ['file_progress']

ROOT = Path(__file__).resolve().parents[1]
FPS = 24


def run(command, *, cwd, data=None):
    command = [str(arg) for arg in command]
    result = subprocess.run(command, input=data, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=cwd, timeout=180,
                            env={**os.environ, 'NO_COLOR': '1'})
    with (Path(cwd) / 'commands.log').open('a') as log:
        log.write(f'\n{command!r}\n{result.stderr.decode(errors="replace")}\n')
        if command[0] == sys.executable:
            log.write(result.stdout.decode(errors='replace'))
    assert result.returncode == 0, (
        f'Command failed ({result.returncode}): {command!r}\n'
        f'{result.stdout[-6000:].decode(errors="replace")}\n'
        f'{result.stderr[-6000:].decode(errors="replace")}')
    return result.stdout


class Media:
    def __init__(self, directory):
        self.directory = directory
        self._probe_cache = {}

    def encode(self, name, frames, *, encoding='h264'):
        """Use viewable H.264 by default; special formats must be requested explicitly."""
        path = self.directory / name
        height, width = frames.shape[1:3]
        codecs = {
            'h264': ['-c:v', 'libx264', '-crf', '10', '-pix_fmt', 'yuv420p',
                     '-vf', 'scale=out_color_matrix=bt601:out_range=tv',
                     '-colorspace', 'smpte170m', '-color_primaries', 'smpte170m',
                     '-color_trc', 'smpte170m', '-color_range', 'tv',
                     '-tag:v', 'avc1', '-movflags', '+faststart+write_colr'],
            'prores10': ['-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le',
                         '-vf', 'scale=out_color_matrix=bt709:out_range=tv',
                         '-colorspace', 'bt709', '-color_primaries', 'bt709',
                         '-color_trc', 'bt709', '-color_range', 'tv',
                         '-movflags', '+write_colr'],
        }
        codec = codecs[encoding]
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
             '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', FPS,
             '-i', 'pipe:0', *codec, path], cwd=self.directory,
            data=np.ascontiguousarray(frames, dtype=np.uint8).tobytes())
        return path

    def mux_audio(self, source, frequencies=(440,), *, unsupported_first=None):
        """Generate all audio. Optional apac-tagged PCM models an undecodable track.

        This is deliberately not an APAC encoder or a copy of personal audio.
        The MOV remains playable through its default AAC track.
        """
        output = self.directory / f'{source.stem}-audio.mov'
        seconds = float(self.probe(source)['streams'][0]['duration'])
        args = ['ffmpeg', '-v', 'error', '-y', '-i', source]
        for frequency in frequencies:
            args += ['-f', 'lavfi', '-i', f'sine=frequency={frequency}:sample_rate=48000:duration={seconds}']
        if unsupported_first is not None:
            args += ['-f', 'lavfi', '-i', f'anullsrc=r=48000:cl=quad:d={seconds}']
        args += ['-map', '0:v:0']
        tracks = list(range(1, len(frequencies) + 1))
        if unsupported_first is not None:
            tracks.insert(0 if unsupported_first else len(tracks), len(frequencies) + 1)
        for track in tracks:
            args += ['-map', f'{track}:a:0']
        args += ['-map_metadata', '-1', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k']
        if unsupported_first is not None:
            index = tracks.index(len(frequencies) + 1)
            args += [f'-c:a:{index}', 'pcm_s16le', f'-disposition:a:{index}', '0']
        args += [output]
        run(args, cwd=self.directory)
        if unsupported_first is not None:
            # Modify only the generated PCM sample-entry type in stsd, not payloads.
            data = bytearray(output.read_bytes())
            changed = []

            def visit(start, end):
                while start < end:
                    size = int.from_bytes(data[start:start+4], 'big')
                    kind = data[start+4:start+8]
                    assert size >= 8 and start + size <= end
                    if kind in (b'moov', b'trak', b'mdia', b'minf', b'stbl'):
                        visit(start + 8, start + size)
                    elif kind == b'stsd':
                        entry = start + 16
                        count = int.from_bytes(data[start+12:start+16], 'big')
                        for _ in range(count):
                            if data[entry+4:entry+8] == b'sowt':
                                data[entry+4:entry+8] = b'apac'
                                changed.append(entry)
                            entry += int.from_bytes(data[entry:entry+4], 'big')
                    start += size

            visit(0, len(data))
            assert len(changed) == 1
            output.write_bytes(data)
        return output

    def audio(self, path):
        raw = run(['ffmpeg', '-v', 'error', '-i', path, '-map', '0:a:0',
                   '-ac', '1', '-ar', '48000', '-f', 'f32le', '-'], cwd=self.directory)
        return np.frombuffer(raw, dtype='<f4')

    def streams(self, path):
        return self._probe(path, ('-show_streams',))['streams']

    def process(self, *args, name='output.mp4'):
        output = self.directory / name
        run([sys.executable, ROOT / 'ffswak.py', '-o', output, *args], cwd=self.directory)
        assert output.is_file(), f'Missing output: {output}'
        # ffswak preserves richer formats; provide a review copy for those outputs.
        video_streams = self.probe(output)['streams']
        if video_streams and video_streams[0]['pix_fmt'] not in ('yuv420p', 'yuv420p10le'):
            self.preview(output)
        return output

    def probe(self, path):
        return self._probe(path, ('-select_streams', 'v:0', '-show_streams', '-show_frames'))

    def _probe(self, path, options):
        path = Path(path)
        path = (self.directory / path).resolve() if not path.is_absolute() else path.resolve()
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        key = (path, options)
        cached = self._probe_cache.get(key)
        if cached is None or cached[0] != signature:
            result = json.loads(run(['ffprobe', '-v', 'error', *options, '-of', 'json', path],
                                    cwd=self.directory))
            self._probe_cache[key] = (signature, result)
        # Preserve the previous behavior: callers can mutate their result without affecting later probes.
        return copy.deepcopy(self._probe_cache[key][1])

    def preview(self, path):
        """Write a player-friendly review copy; assertions still use the original."""
        output = path.with_name(f'{path.stem}-preview.mp4')
        run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-map', '0:v:0',
             '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
             '-tag:v', 'avc1', '-movflags', '+faststart', output], cwd=self.directory)
        return output

    def decode(self, path):
        info = self.probe(path)
        stream = info['streams'][0]
        raw = run(['ffmpeg', '-v', 'error', '-i', path, '-map', '0:v:0',
                   '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                  cwd=self.directory)
        return np.frombuffer(raw, np.uint8).reshape(-1, stream['height'], stream['width'], 3)


@pytest.fixture
def media(tmp_path):
    for executable in ('ffmpeg', 'ffprobe'):
        if not shutil.which(executable):
            pytest.fail(f'{executable} is required; see tests/README.md')
    return Media(tmp_path)


@pytest.fixture
def stabilization_available(media):
    filters = run(['ffmpeg', '-hide_banner', '-filters'], cwd=media.directory).decode()
    if not all(name in filters for name in ('vidstabdetect', 'vidstabtransform')):
        pytest.fail('Stabilization tests require FFmpeg built with libvidstab')
