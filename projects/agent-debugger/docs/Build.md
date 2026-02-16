# Build & Release

## CI

Pushing to `main` triggers the `check_diffs.yml` workflow which runs lint and tests on Python 3.11 and 3.12.

## Publishing to PyPI

Publishing is triggered by pushing a version tag.

**Important:** Bump `version` in `pyproject.toml` before tagging. PyPI rejects uploads if a wheel with the same filename already exists, and the filename includes the version number.

```bash
# 1. Bump version in pyproject.toml
# 2. Commit the version bump
git add projects/agent-debugger/pyproject.toml
git commit -m "Bump version to 0.1.2"

# 3. Tag and push
git tag v0.1.2
git push origin main v0.1.2
```

If you tagged without bumping the version, delete and re-push the tag after fixing:

```bash
git tag -d v0.1.2
git push origin :refs/tags/v0.1.2
# bump version, commit, then re-tag
git tag v0.1.2
git push origin main v0.1.2
```

The `release.yml` workflow will:

1. Build the distribution artifacts (`uv build`)
2. Publish to [TestPyPI](https://test.pypi.org/p/agent-debugger)
3. Publish to [PyPI](https://pypi.org/p/agent-debugger)

Both publish jobs require their respective GitHub environments (`testpypi` / `pypi`) to be configured with trusted publisher (OIDC) permissions.

## Optional Dependencies

To enable the LiteLLM path in the example agent:

```bash
uv pip install -e ".[litellm]"
```

This installs `python-dotenv` and `langchain-litellm`. If using a Vertex AI model (e.g. `vertex_ai/gemini-2.5-flash-lite`), also install:

```bash
uv pip install google-cloud-aiplatform
```
