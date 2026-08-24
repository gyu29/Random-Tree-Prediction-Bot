"""Coverage for app/model_registry.py's save/load contract: load() must verify the
manifest signature and every artifact's sha256 before trusting a .pkl, and reject --
not silently ignore -- any mismatch. This is the one piece of the app whose whole job
is to detect tampering/corruption, so these tests exercise that path directly rather
than relying on it only ever being exercised incidentally by other tests.

isolated_registry (below) points MODELS_ROOT at a throwaway tmp_path and fakes the
signing key for every test in this file, so nothing here can ever touch the real
models/ directory or write a MODEL_SIGNING_KEY into the real .env.
"""
import json
import os
import sys

import joblib
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import model_registry  # noqa: E402

TEST_SIGNING_KEY = "ab" * 32  # same shape as os.urandom(32).hex()


class _FakeModel:
    """Plain, joblib-picklable stand-in for a trained model. save()/load() only care
    that artifacts round-trip through joblib, not what kind of object they hold."""

    def __init__(self, decision_threshold=0.65):
        self.decision_threshold = decision_threshold


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "MODELS_ROOT", str(tmp_path))
    monkeypatch.setattr(
        model_registry.SecretManager,
        "get_or_create_signing_key",
        staticmethod(lambda: TEST_SIGNING_KEY),
    )


def _save_fixture(category="widgets", **overrides):
    defaults = dict(
        model=_FakeModel(),
        scaler={"kind": "fake-scaler"},
        feature_columns=["close", "rsi_14"],
        training_stats={"decision_threshold": 0.65, "trained_rows": 123},
        dataset_snapshot=[
            {"file": "AAPL.csv", "sha256": "deadbeef", "rows": 500,
             "date_min": "2020-01-01", "date_max": "2021-01-01"}
        ],
        hyperparameters={"n_estimators": 50, "decision_threshold": 0.65},
    )
    defaults.update(overrides)
    return model_registry.save(category, **defaults)


def test_save_then_load_roundtrips_all_artifacts():
    _save_fixture("widgets", model=_FakeModel(decision_threshold=0.65))

    loaded = model_registry.load("widgets")

    assert loaded.model.decision_threshold == 0.65
    assert loaded.scaler == {"kind": "fake-scaler"}
    assert loaded.feature_columns == ["close", "rsi_14"]
    assert loaded.training_stats == {"decision_threshold": 0.65, "trained_rows": 123}
    assert loaded.manifest["category"] == "widgets"
    assert loaded.manifest["hyperparameters"]["n_estimators"] == 50
    assert loaded.version_warnings == []


def test_is_trained_and_list_trained_categories_reflect_disk_state():
    assert model_registry.is_trained("widgets") is False
    assert model_registry.list_trained_categories() == []

    _save_fixture("widgets")

    assert model_registry.is_trained("widgets") is True
    assert model_registry.list_trained_categories() == ["widgets"]


def test_is_trained_is_false_when_an_artifact_is_missing():
    _save_fixture("widgets")
    os.remove(os.path.join(model_registry.category_dir("widgets"), "scaler.pkl"))

    assert model_registry.is_trained("widgets") is False


def test_load_missing_category_raises_not_found():
    with pytest.raises(model_registry.ModelNotFoundError):
        model_registry.load("never_trained")


def test_load_missing_signature_file_raises_not_found_not_integrity_error():
    """A missing manifest/signature means "never saved" (ModelNotFoundError), distinct
    from present-but-wrong (ModelIntegrityError, covered below) -- load() must not
    conflate the two."""
    _save_fixture("widgets")
    os.remove(os.path.join(model_registry.category_dir("widgets"), "manifest.sig"))

    with pytest.raises(model_registry.ModelNotFoundError):
        model_registry.load("widgets")


def test_load_rejects_hand_edited_manifest():
    _save_fixture("widgets")
    manifest_path = os.path.join(model_registry.category_dir("widgets"), "manifest.json")
    with open(manifest_path, "r+", encoding="utf-8") as handle:
        manifest = json.load(handle)
        manifest["hyperparameters"]["n_estimators"] = 999  # edited without re-signing
        handle.seek(0)
        json.dump(manifest, handle)
        handle.truncate()

    with pytest.raises(model_registry.ModelIntegrityError, match="signature mismatch"):
        model_registry.load("widgets")


def test_load_rejects_manifest_signed_with_a_different_key(monkeypatch):
    _save_fixture("widgets")

    monkeypatch.setattr(
        model_registry.SecretManager, "get_or_create_signing_key",
        staticmethod(lambda: "ff" * 32),
    )

    with pytest.raises(model_registry.ModelIntegrityError, match="signature mismatch"):
        model_registry.load("widgets")


def test_load_rejects_tampered_artifact_bytes():
    _save_fixture("widgets")
    scaler_path = os.path.join(model_registry.category_dir("widgets"), "scaler.pkl")
    with open(scaler_path, "ab") as handle:
        handle.write(b"tampered")  # changes the sha256 without touching the manifest

    with pytest.raises(model_registry.ModelIntegrityError, match="scaler.pkl"):
        model_registry.load("widgets")


def test_load_rejects_missing_artifact_file():
    _save_fixture("widgets")
    os.remove(os.path.join(model_registry.category_dir("widgets"), "features.pkl"))

    with pytest.raises(model_registry.ModelIntegrityError, match="Missing artifact 'features.pkl'"):
        model_registry.load("widgets")


def test_save_leaves_no_partial_state_on_failure():
    """model=lambda can't be pickled, so save() must fail before the manifest/signature
    are written -- and its except/raise around the staging dir must not leave a half
    -written category directory (or an orphaned staging dir) behind for load() to trip over."""
    with pytest.raises(Exception):
        model_registry.save(
            "broken", model=lambda x: x, scaler={}, feature_columns=[],
            training_stats={}, dataset_snapshot=[], hyperparameters={},
        )

    assert not os.path.isdir(model_registry.category_dir("broken"))
    assert os.listdir(model_registry.MODELS_ROOT) == []


def test_update_decision_threshold_roundtrips_and_stays_loadable():
    _save_fixture("widgets", model=_FakeModel(decision_threshold=0.65))

    model_registry.update_decision_threshold("widgets", 0.42)
    loaded = model_registry.load("widgets")

    assert loaded.training_stats["decision_threshold"] == 0.42
    assert loaded.manifest["hyperparameters"]["decision_threshold"] == 0.42
    assert loaded.model.decision_threshold == 0.42


def test_update_decision_threshold_on_untrained_category_raises_not_found():
    with pytest.raises(model_registry.ModelNotFoundError):
        model_registry.update_decision_threshold("never_trained", 0.5)
