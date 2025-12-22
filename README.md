# Maternal Instinct AI — Docker & CI

This repository runs a Streamlit app. The project includes a `Dockerfile` and a GitHub Actions workflow to build and publish a container image to GitHub Container Registry (GHCR).

Quick local build & run

```powershell
docker build -t maternal-instinct-ai:latest .
docker run --rm -p 8501:8501 maternal-instinct-ai:latest
```

CI: GitHub Actions

- Workflow file: [.github/workflows/docker-build-publish.yml](.github/workflows/docker-build-publish.yml)
- The workflow builds multi-arch images and pushes to Docker Hub as `${DOCKERHUB_USERNAME}/maternal-instinct-ai:latest` and a commit-sha tag.

Requirements for publishing
- Set up secrets in your GitHub repository: `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (your Docker Hub access token).
- Ensure repository `Workflows permissions` allow `Read and write permissions` for `Contents` and `Packages`.

If you want, I can:
- Update the workflow to push to Docker Hub instead.
- Add tagging based on Git tags or releases.
- Create a small test that runs `python -m pytest` in CI before building the image.
