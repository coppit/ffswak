# Contributing to ffswak

The source repository is [github.com/coppit/ffswak](https://github.com/coppit/ffswak). Please report bugs and propose
enhancements in its [issue tracker](https://github.com/coppit/ffswak/issues).

## Development setup

ffswak requires Python 3.11 or newer, plus `ffmpeg` and `ffprobe` on `PATH`. The media tests also need an FFmpeg build
with libx265, libx264, libvidstab, and ProRes encoding support.

Clone the repository, install [uv](https://docs.astral.sh/uv/), and create the development environment:

```sh
git clone https://github.com/coppit/ffswak.git
cd ffswak
uv sync --group test --group lint
```

Run the checks before submitting a change:

```sh
uv run pytest
uv run ruff check .
```

The test suite generates small synthetic videos, invokes the actual command-line program, and inspects the result. See
[tests/README.md](tests/README.md) for test behavior, focused runs, review artifacts, fixtures, and guidance on adding a
test.

Keep code and documentation lines to 120 characters or fewer.
