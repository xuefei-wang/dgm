import io
import tarfile
import importlib
import sys
import types

import pytest

if "docker" not in sys.modules:
    docker_module = types.ModuleType("docker")
    errors_module = types.ModuleType("docker.errors")

    class _DockerError(Exception):
        pass

    class NotFound(_DockerError):
        pass

    class APIError(_DockerError):
        pass

    errors_module.NotFound = NotFound
    errors_module.APIError = APIError
    docker_module.errors = errors_module
    sys.modules["docker"] = docker_module
    sys.modules["docker.errors"] = errors_module

from utils import docker_utils

from utils import docker_utils


def _build_tar_for_members(members):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for name, payload in members.items():
            encoded = payload.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))
    stream.seek(0)
    return stream


def test_docker_utils_extract_rejects_path_traversal(tmp_path):
    stream = _build_tar_for_members({"../evil.txt": "pwned"})
    with tarfile.open(fileobj=stream, mode="r") as tar:
        with pytest.raises(ValueError, match="Unsafe tar member path"):
            docker_utils._extract_tar_to_directory(tar, destination=tmp_path)


def test_docker_utils_and_swe_bench_share_safe_extraction(tmp_path):
    stream = _build_tar_for_members({"subdir/good.txt": "ok"})
    with tarfile.open(fileobj=stream, mode="r") as docker_tar:
        docker_utils._extract_tar_to_directory(docker_tar, destination=tmp_path)
    assert (tmp_path / "subdir" / "good.txt").read_text() == "ok"

    stream = _build_tar_for_members({"nested/ok.txt": "yes"})
    sweep_root = tmp_path / "sweep"
    sweep_root.mkdir()
    swe_bench_utils = importlib.import_module("swe_bench.utils")

    with tarfile.open(fileobj=stream, mode="r") as swe_tar:
        swe_bench_utils._extract_tar_to_directory(swe_tar, destination=sweep_root)
    assert (sweep_root / "nested" / "ok.txt").read_text() == "yes"
