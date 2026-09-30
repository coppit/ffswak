# Maintainer notes

The canonical repository is [github.com/coppit/ffswak](https://github.com/coppit/ffswak). Bugs and feature requests are
filed in its [issue tracker](https://github.com/coppit/ffswak/issues).

## Publishing a release

The `Package` GitHub Actions workflow runs the full test suite and Ruff, builds the distribution, and installs its wheel
in a clean environment on every push and pull request. It installs FFmpeg on Ubuntu 24.04; the tests require
stabilization support and fail if it is missing. These checks do not publish anything.

The `Publish to PyPI` workflow builds and publishes when a tag named `v<version>` is pushed. It refuses to publish when
the tag does not match the version in `pyproject.toml`. It calls the same `Package` workflow and requires all checks to
pass before publishing. Merely changing the version in `pyproject.toml` and pushing a branch does not publish a release;
you must push the Git tag separately. The project's PyPI trusted publisher is configured for `coppit/ffswak`, workflow
`publish.yml`, and environment `pypi`. Publishing requires approval in that GitHub environment.

## Manual package test

The GitHub Actions workflow performs this check automatically. Run it locally before a release when you want to inspect
the exact package that will be uploaded:

```sh
uv build --no-sources
smoke_dir="$(mktemp -d)"
uv venv "$smoke_dir" --python 3.11
uv pip install --python "$smoke_dir/bin/python" dist/*.whl
"$smoke_dir/bin/ffswak" --help
rm -rf "$smoke_dir"
```

This builds both the source distribution and wheel, then installs the wheel in an otherwise empty environment. The help
command confirms that the installed console script starts; normal operation also requires `ffmpeg` and `ffprobe`.

## Release checklist

1. Update the version in `pyproject.toml`.
2. Install the development environment and run the checks:

   ```sh
   uv sync --group test --group lint
   uv run pytest
   uv run ruff check .
   ```

3. Run the manual package test above if you want to inspect the release artifact locally.
4. Commit the version change and confirm that the `Package` workflow passes on GitHub.
5. Create and push the matching tag. For example, version `0.1.0` uses tag `v0.1.0`:

   ```sh
   git tag v0.1.0
   git push origin v0.1.0
   ```

6. Approve the `pypi` GitHub environment, if it requires review. The `Publish to PyPI` workflow then uploads
   the package.
7. Confirm the new version on PyPI and test it without a persistent installation:

   ```sh
   uvx --from 'ffswak==0.1.0' ffswak --help
   ```

   Replace `0.1.0` with the version just published.

## Documentation updates

Commit and push documentation changes to update the GitHub README and guides. No version bump or release tag is needed.
PyPI displays the README bundled with the published release, so updating its description requires a new release.
A documentation-only patch release such as `0.1.1` is sufficient; there is no need to jump to `0.2.0`.

## Repository links in package metadata

The `Repository` and `Issues` links in `pyproject.toml` are shown by package indexes. Keep them pointed at the canonical
GitHub repository and its issue tracker.
