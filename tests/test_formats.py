"""Practical media regressions for bit depth, range and metadata fallbacks."""
import numpy as np
import pytest
from conftest import FPS, run
from test_video import quadrants, assert_timeline

pytestmark = pytest.mark.integration


@pytest.mark.parametrize('codec,pixel_format', [('libx265', 'yuv420p10le'), ('prores_ks', 'yuv422p10le')])
def test_homogeneous_10bit_video_preserves_depth_and_gradient(media, codec, pixel_format):
    # 1edcceb: preserving homogeneous formats must not quietly select 8-bit output.
    # Construct the gradient in 10-bit YUV directly, not by expanding 8-bit RGB.
    source = media.directory / 'gradient.mov'
    extra = ['-tag:v', 'hvc1', '-x265-params', 'lossless=1'] if codec == 'libx265' else ['-profile:v', '3']
    run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
         f'nullsrc=s=640x96:r={FPS}:d=2,format={pixel_format},geq=lum=64+X*1.3:cb=512:cr=512',
         '-c:v', codec, '-pix_fmt', pixel_format, *extra, source], cwd=media.directory)
    # Reverse forces processing rather than the no-edit copy fallback.
    output = media.process('-R', source)
    assert_timeline(media, output, 2, (640, 96))
    assert media.probe(output)['streams'][0]['pix_fmt'] == pixel_format

    def luma10(path):
        data = run(['ffmpeg', '-v', 'error', '-i', path, '-frames:v', '1',
                    '-pix_fmt', 'yuv420p10le', '-f', 'rawvideo', '-'], cwd=media.directory)
        return np.frombuffer(data, '<u2')[:640*96].reshape(96, 640)

    expected, actual = luma10(source), luma10(output)
    assert len(np.unique(expected)) > 256
    assert len(np.unique(actual)) > 256
    assert np.sqrt(np.mean((actual.astype(float) - expected) ** 2)) < 8
    assert np.percentile(actual, 99) - np.percentile(actual, 1) > 750


def test_full_range_mjpeg_preserves_black_white_and_midtones(media):
    # A camera-style full-range MJPEG MOV is playable and exercises yuvj formats.
    ramp = np.linspace(0, 255, 320).astype(np.uint8)
    frames = np.broadcast_to(ramp[None, None, :, None], (2*FPS, 240, 320, 3))
    h264 = media.encode('gradient.mov', frames)
    source = media.directory / 'full-range.mov'
    run(['ffmpeg', '-v', 'error', '-i', h264, '-c:v', 'mjpeg', '-q:v', '2',
         '-pix_fmt', 'yuvj420p', source], cwd=media.directory)
    assert media.probe(source)['streams'][0]['color_range'] == 'pc'
    output = media.process('-R', source)
    reference = media.decode(source).astype(float)
    actual = media.decode(output).astype(float)
    assert actual.shape == reference.shape
    assert np.quantile(abs(actual-reference), .99) < 12
    assert actual[:, :, :8].mean() < 8
    assert actual[:, :, -8:].mean() > 247


def test_playable_mkv_without_stream_bitrate_uses_container_fallback(media):
    # f6a93ca specifically addressed MKV lacking per-stream bit_rate.
    source = media.encode('scene.mov', quadrants())
    mkv = media.directory / 'scene.mkv'
    run(['ffmpeg', '-v', 'error', '-i', source, '-c', 'copy', mkv], cwd=media.directory)
    assert 'bit_rate' not in media.probe(mkv)['streams'][0]
    output = media.process('-R', mkv)
    assert_timeline(media, output, 2, (320, 240))
    assert np.quantile(abs(media.decode(output).astype(float) - media.decode(source)), .99) < 18
