from pathlib import Path

from aurarouter.models.file_storage import FileModelStorage


def test_register_and_list(tmp_path):
    """Register a model and verify it appears in list_models."""
    storage = FileModelStorage(tmp_path)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake-gguf")

    storage.register(repo="Qwen/Test", filename="model.gguf", path=model_file)

    models = storage.list_models()
    assert len(models) == 1
    assert models[0]["filename"] == "model.gguf"
    assert models[0]["repo"] == "Qwen/Test"
    assert models[0]["size_bytes"] == len(b"fake-gguf")
    assert models[0]["path"] == str(model_file)


def test_has_model(tmp_path):
    """has_model returns True for registered models, False otherwise."""
    storage = FileModelStorage(tmp_path)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"data")

    storage.register(repo="org/repo", filename="model.gguf", path=model_file)

    assert storage.has_model("model.gguf") is True
    assert storage.has_model("nonexistent.gguf") is False


def test_has_model_unregistered_but_on_disk(tmp_path):
    """has_model detects files that exist on disk even if not registered."""
    storage = FileModelStorage(tmp_path)
    model_file = tmp_path / "orphan.gguf"
    model_file.write_bytes(b"data")

    assert storage.has_model("orphan.gguf") is True


def test_get_model_path(tmp_path):
    """get_model_path resolves registered filenames to full paths."""
    storage = FileModelStorage(tmp_path)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"data")

    storage.register(repo="org/repo", filename="model.gguf", path=model_file)

    assert storage.get_model_path("model.gguf") == model_file
    assert storage.get_model_path("missing.gguf") is None


def test_remove(tmp_path):
    """remove removes from registry but keeps the file by default."""
    storage = FileModelStorage(tmp_path)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"data")

    storage.register(repo="org/repo", filename="model.gguf", path=model_file)
    assert storage.remove("model.gguf") is True
    assert storage.list_models() == []
    assert model_file.is_file()  # File still on disk


def test_remove_with_delete(tmp_path):
    """remove with delete_file=True removes from registry AND deletes the file."""
    storage = FileModelStorage(tmp_path)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"data")

    storage.register(repo="org/repo", filename="model.gguf", path=model_file)
    assert storage.remove("model.gguf", delete_file=True) is True
    assert storage.list_models() == []
    assert not model_file.is_file()  # File deleted


def test_remove_nonexistent(tmp_path):
    """remove returns False for models not in registry."""
    storage = FileModelStorage(tmp_path)
    assert storage.remove("missing.gguf") is False


def test_scan_finds_unregistered(tmp_path):
    """scan discovers .gguf files in the directory that aren't registered."""
    storage = FileModelStorage(tmp_path)

    # Place files directly (not via register)
    (tmp_path / "a.gguf").write_bytes(b"aaa")
    (tmp_path / "b.gguf").write_bytes(b"bbb")
    (tmp_path / "readme.txt").write_bytes(b"not a model")

    added = storage.scan()
    assert added == 2
    filenames = {m["filename"] for m in storage.list_models()}
    assert filenames == {"a.gguf", "b.gguf"}


def test_scan_skips_already_registered(tmp_path):
    """scan doesn't duplicate already-registered models."""
    storage = FileModelStorage(tmp_path)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"data")
    storage.register(repo="org/repo", filename="model.gguf", path=model_file)

    added = storage.scan()
    assert added == 0
    assert len(storage.list_models()) == 1


def test_persistence(tmp_path):
    """Registry survives creating a new FileModelStorage instance."""
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"data")

    storage1 = FileModelStorage(tmp_path)
    storage1.register(repo="org/repo", filename="model.gguf", path=model_file)

    # New instance should load persisted registry
    storage2 = FileModelStorage(tmp_path)
    models = storage2.list_models()
    assert len(models) == 1
    assert models[0]["filename"] == "model.gguf"


def test_register_updates_existing(tmp_path):
    """Registering the same filename again updates the entry."""
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"data")

    storage = FileModelStorage(tmp_path)
    storage.register(repo="org/v1", filename="model.gguf", path=model_file)
    storage.register(repo="org/v2", filename="model.gguf", path=model_file)

    models = storage.list_models()
    assert len(models) == 1
    assert models[0]["repo"] == "org/v2"


def test_default_dir():
    """Default models_dir is ~/.auracore/models/."""
    storage = FileModelStorage()
    assert storage.models_dir == Path.home() / ".auracore" / "models"
