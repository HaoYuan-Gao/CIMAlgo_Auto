import io
import subprocess
import tarfile

import pytest

import utils.archive_tool as archive_tools


@pytest.fixture
def package(tmp_path):
    archive = tmp_path / "SDK_v1.tar.gz"
    install_dir = tmp_path / "installed"

    def build(files, *, links=None):
        with tarfile.open(archive, "w:gz") as output:
            for name in files:
                member = tarfile.TarInfo(name)
                member.size = 2
                output.addfile(member, io.BytesIO(b"ok"))
            for name, target in (links or {}).items():
                member = tarfile.TarInfo(name)
                member.type = tarfile.SYMTYPE
                member.linkname = target
                output.addfile(member)
        return archive, install_dir

    return build


def test_python_rejects_symlink_outside_content_without_wrapper(package, tmp_path):
    archive, _ = package(
        ["safe/file"],
        links={"safe/link": "../../outside"},
    )

    with pytest.raises(archive_tools.ArchiveError):
        archive_tools.extract(archive, tmp_path / "content")


def test_deprecated_tar_uses_literal_anchored_exclusions(tmp_path, monkeypatch):
    commands = []

    def record(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(archive_tools.subprocess, "run", record)
    with pytest.warns(DeprecationWarning, match="tar 指令"):
        archive_tools.extract_with_tar(
            tmp_path / "package.tar.gz",
            tmp_path / "content",
            "SDK",
            exclude_dirs=("Host/cache",),
        )

    extract_command = commands[-1]
    assert "--anchored" in extract_command
    assert "--no-wildcards" in extract_command
    assert "--exclude=SDK/Host/cache" in extract_command
    assert "--exclude=Host/cache" not in extract_command


def test_keeps_wrapper_without_explicit_configuration(package, tmp_path):
    archive, _ = package(["BigRoot/Host/file", "BigRoot/Agent/file"])

    extracted = archive_tools.extract(archive, tmp_path / "content")

    assert (extracted / "BigRoot/Host/file").is_file()


def test_rejects_member_outside_configured_wrapper(package, tmp_path):
    archive, _ = package(["SDK/file", "README"])

    with pytest.raises(archive_tools.ArchiveError, match="包装目录外"):
        archive_tools.extract(
            archive,
            tmp_path / "content",
            wrapper_dir="SDK",
        )


def test_rejects_unsafe_member_path_before_extraction(package, tmp_path):
    archive, _ = package(["Safe/file", "Other/file", "../outside"])

    with pytest.raises(archive_tools.ArchiveError, match="不安全路径"):
        archive_tools.extract(archive, tmp_path / "content")

    assert not (tmp_path / "outside").exists()


def test_archive_functions_compress_and_extract_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.bin").write_bytes(b"ok")
    archive = tmp_path / "package.tar.gz"

    assert archive_tools.compress(archive, source) == archive
    extracted = archive_tools.extract(
        archive,
        tmp_path / "content",
        wrapper_dir=source.name,
    )

    assert (extracted / "data.bin").read_bytes() == b"ok"


def test_compresses_directory_contents_without_root_directory(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "data.bin").write_bytes(b"data")
    (source / "nested/file.bin").write_bytes(b"nested")
    archive = tmp_path / "package.tar.gz"

    archive_tools.compress(archive, source, include_root=False)
    extracted = archive_tools.extract(archive, tmp_path / "content")

    assert not (extracted / source.name).exists()
    assert (extracted / "data.bin").read_bytes() == b"data"
    assert (extracted / "nested/file.bin").read_bytes() == b"nested"


def test_compress_rejects_non_directory_source(tmp_path):
    source = tmp_path / "file.bin"
    source.write_bytes(b"data")

    with pytest.raises(archive_tools.ArchiveError, match="不是普通目录"):
        archive_tools.compress(tmp_path / "package.tar.gz", source)
