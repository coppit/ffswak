# Video regression tests

Use Python 3.10+ and pytest 9.1.1 (the latest stable release when this harness was added). The application itself does
not acquire this newer Python requirement.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest
```

The suite runs four tests concurrently using pytest-xdist. Each test retains its own temporary files and command
logs, and all media assertions still run. Four workers keep concurrency bounded because FFmpeg also uses threads.
Install the updated `requirements-test.txt` in an existing environment before running the suite.

If you use an existing `python3` environment instead of `.venv`, install dependencies with that same interpreter:

```sh
python3 -m pip install -r requirements-test.txt
python3 -m pytest
```

An `unrecognized arguments: -n` error means pytest-xdist is missing from the environment running pytest. Run the
installation command above; installing it in a different virtual environment will not fix that interpreter.

Use `python -m pytest -n 0` for sequential execution or debugging, or `python -m pytest -n 2` to reduce CPU and memory
use. For a focused test, `python -m pytest -n 0 tests/test_metadata_errors.py` avoids worker startup overhead.
On the development machine, all 130 tests passed in about 23 seconds with four workers, compared with 82 seconds
sequentially. Timing depends on the machine and FFmpeg build.

Interactive runs show a live progress row for each test file. Each row includes a bar and completed/total counts;
`F` counts tests with failures or setup/teardown errors, and `s/x` counts skipped or expected-failure tests. Completion
is counted after teardown. Pytest still prints its usual failure details and final result summary.
Use `--file-progress=off` for standard pytest output, or `--file-progress=on` to force the display. Automatic mode
keeps standard output when redirected, when using `-v`, or when capture is disabled with `-s`.

## Developer checks

Ruff checks for undefined names and unnecessary `global` declarations. It is a source check, separate from the media
regression suite:

```sh
.venv/bin/python -m ruff check .
```

FFmpeg and ffprobe must be on PATH. FFmpeg needs libx265, libx264, libvidstab, ProRes encoding. Missing binaries or
stabilization filters fail explicitly; the suite does not silently skip the main integration tests. Tested with FFmpeg
8.0.  Media integration tests invoke the actual CLI and decode its output. Focused tests also exercise malformed ffprobe
records and subprocess failures directly, using controlled metadata or small child processes. No downloads, personal
videos, or personal videos are required. Interlacing detection also uses checked-in GPL-3.0 MKV fixtures; see
`fixtures/interlacing/SOURCE.md` for their source and license.

## How the harness works

Pytest discovers the test functions in `test_*.py` and supplies the `media` fixture from `conftest.py`. Each test gets
its own temporary directory and a `Media` helper that operates there. The tests define the scene and expected result;
the helper handles video files and subprocesses.

Each `Media` helper caches probe results within its test. Repeated metadata/frame requests for an unchanged file
reuse the result; different probe options stay separate. File identity, size, and modification/change timestamps
invalidate cached results when inputs or outputs change. Callers receive independent copies of the metadata.
The cache is discarded after each test and requires no shared files or worker synchronization.

The data flows through four steps:

1. **Generate inputs.** A test builds a NumPy array of RGB frames with known content. `Media.encode()` pipes those
pixels into FFmpeg to create an actual input video. H.264 with 8-bit YUV 4:2:0 is the default so ordinary fixtures are
directly viewable. Tests that need higher bit depth explicitly request 10-bit ProRes in MOV. These input formats are
intended for ordinary media players.  Frame arrays have shape `(frames, height, width, 3)`; their order and the fixture
frame rate determine the input timeline.
2. **Run the command.** `Media.process()` launches `ffswak.py` in a separate process using the same Python interpreter
as pytest. It passes the test's CLI arguments and an explicit output path in the temporary directory. This exercises
argument parsing and the real FFmpeg pipeline. The subprocess helper checks exit status, applies a timeout, and records
commands and diagnostics.
3. **Read the result.** `Media.probe()` uses ffprobe to read stream metadata and frame timestamps. `Media.decode()` uses
FFmpeg to decode every video frame to RGB24, returning another NumPy array. It preserves the output frame sequence
without inserting or dropping frames to impose a new frame rate. Frames stay in memory rather than being saved as
individual images.
4. **Assert behavior.** Tests inspect metadata, timing, and relevant regions of the decoded frames. Expected values come
from the known input scene and the requested operation, rather than from another invocation of ffswak. Color checks
examine patches across frames; motion checks track a landmark across the sequence. Decoding every frame does not mean
asserting every pixel: each test chooses the measurements that establish its intended behavior.

To add a test, define a recognizable input, encode it with `media.encode()`, run the desired arguments with
`media.process()`, and inspect the result with `media.probe()` and/or `media.decode()`. Keep the behavior in the test
name and explain the scene, expected result, and any excluded regions in nearby comments.  Small, short videos keep
these integration tests fast and memory use bounded.

## Interlacing fixtures

`fixtures/interlacing/` contains upstream MKV patterns for TFF, BFF, progressive, hard-telecined, and soft-telecined
video. Their visual field and cadence markers make interlacing visible during manual review. The tests run FFmpeg's
`idet` over the actual files and verify the resulting classification. The source repository, GPL-3.0 license, and
generator script are retained alongside the media.

## Fuzzy comparisons and review artifacts

Encoded files are intentionally **not** compared byte-for-byte. Each checked color patch must have 99% of channel errors
within 18/255 on **every frame**.  Patches avoid color edges, where chroma subsampling and ringing are expected.  These
limits tolerate codec noise while rejecting wrong quadrants, missing clips, wrong time ranges and material color shifts.
Stabilization uses a physical motion measurement rather than expecting identical pixels or crop decisions.

Pytest retains recent temporary directories. Each test directory contains its inputs, output.mp4, and commands.log;
stabilization also writes motion.json when landmark tracking succeeds. Use `pytest --basetemp=/tmp/ffswak-review` for a
known review location (pytest **clears that directory** at the next run). Do not point `--basetemp` at a directory
containing anything you want to keep.

Human-certified real-world clips can supplement these synthetic tests later.  For those, retain the source, exact
command, approved output and tool versions; compare aligned decoded frames using per-frame SSIM plus timing/dimension
checks.  Choose thresholds from reviewed examples, and include deliberately broken results as negative controls. Do not
auto-approve a new baseline or use one whole-video average that can hide a bad segment. No certified baseline is needed
for the current tests.

## Inspecting stabilization output

Run the stabilization cases with a predictable artifact location:

```sh
python3 -m pytest -v -k stabilization --basetemp=/tmp/ffswak-review
```

Open the per-test subdirectory under `/tmp/ffswak-review` and compare `shaky.mov` with `output.mp4`. The input is H.264,
and the output is HEVC. `commands.log` records the commands and application output. `motion.json` is written if landmark
tracking succeeds. The videos remain available when an assertion fails. Copy any artifacts you want to keep before
rerunning: pytest clears the chosen base directory at the next run. Omit `-k stabilization` to run the entire suite.

Some tests produce richer HEVC output formats that media players may not support.  For these, the helper also writes
`output-preview.mp4` in H.264 4:2:0. Assertions always inspect the original output; previews are lossy review copies.

## Synthetic unsupported-audio fixture

The stream-selection regression uses a generated MOV with H.264 video, an AAC sine-wave track and an extra four-channel
silent PCM track. The helper changes only that extra track's MOV sample-entry tag to `apac`. The AAC/video tracks remain
ordinary playable media. This models the unrecognized additional audio stream observed in an iPhone MOV without copying
any personal media, metadata or packets.  It is not a valid APAC encoding and does not test APAC decoding. A negative
control verifies that decoding all its audio tracks fails; ffswak must instead select AAC.

Both audio-track orders are exercised. If a future FFmpeg build recognizes the synthetic tag, review the fixture rather
than silently skipping the assertions.  A true APAC interoperability test would require a separately generated or
sanitized valid APAC sample; this suite has no dependency on the original personal file.
