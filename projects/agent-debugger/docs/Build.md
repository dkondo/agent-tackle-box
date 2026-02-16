# Build & Release

## CI

Pushing to `main` triggers the `check_diffs.yml` workflow which runs lint and tests on Python 3.11 and 3.12.

## Publishing to PyPI

Publishing is triggered by pushing a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
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
