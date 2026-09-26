from pathlib import Path


def test_package_files_exist() -> None:
    assert Path("app/main.py").is_file()
    assert Path("app/worker.py").is_file()
    assert Path("compose.hostinger.yaml").is_file()
