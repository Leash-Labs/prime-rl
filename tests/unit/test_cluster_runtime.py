import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_cluster_runtime_is_cuda_13_2_source_build():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    with (ROOT / "uv.lock").open("rb") as handle:
        lock = tomllib.load(handle)

    dependencies = set(project["project"]["dependencies"])
    assert "torch==2.12.1" in dependencies
    assert "transformers==5.6.2" in dependencies
    assert "vllm==0.24.0" in dependencies

    sources = project["tool"]["uv"]["sources"]
    assert sources["torch"]["index"] == "pytorch-cu132"
    assert sources["vllm"]["git"] == "https://github.com/vllm-project/vllm.git"
    assert sources["vllm"]["tag"] == "v0.24.0"

    packages = lock["package"]
    torch = next(package for package in packages if package["name"] == "torch")
    vllm = next(package for package in packages if package["name"] == "vllm")
    assert torch["version"] == "2.12.1+cu132"
    assert "github.com/vllm-project/vllm.git" in vllm["source"]["git"]
