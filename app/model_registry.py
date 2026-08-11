"""Versioned, integrity-checked model artifacts, one set per factor category.

Each models/<category>/ directory holds the joblib artifacts plus two files that
make the model reproducible and tamper-evident:

  manifest.json  Human-readable: library versions, full hyperparameters, a dataset
                 snapshot (hash + row count + date range of every training file
                 used), a code version, timestamp, and a sha256 per artifact file.
  manifest.sig   HMAC-SHA256 over the canonical manifest JSON, keyed by a local
                 MODEL_SIGNING_KEY (auto-generated into .env on first save).

Threat model: this is tamper-evidence for a local single-user research tool --
it detects accidental corruption (e.g. a bad copy, a pandas/sklearn version skew
silently breaking unpickling) and casual tampering by anyone without the local
.env key. It is not a substitute for a real code-signing PKI, and isn't trying
to be one.
"""
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn

from app.config import MODELS_ROOT, PROJECT_ROOT
from app.security import SecretManager

try:
    import xgboost
    _XGBOOST_VERSION = xgboost.__version__
except ImportError:
    _XGBOOST_VERSION = None

ARTIFACT_FILENAMES = {
    "model": "model.pkl",
    "scaler": "scaler.pkl",
    "features": "features.pkl",
    "training_stats": "training_stats.pkl",
}
MANIFEST_FILENAME = "manifest.json"
SIGNATURE_FILENAME = "manifest.sig"


class ModelIntegrityError(Exception):
    """Raised when a manifest signature or artifact hash doesn't match on load."""


class ModelNotFoundError(Exception):
    """Raised when a category has never been trained/saved."""


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sign_manifest(manifest):
    key = bytes.fromhex(SecretManager.get_or_create_signing_key())
    return hmac.new(key, _canonical_json(manifest), hashlib.sha256).hexdigest()


def _library_versions():
    return {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": _XGBOOST_VERSION,
        "joblib": joblib.__version__,
    }


def _code_version():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            return {"type": "git_commit", "value": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        pass

    # No usable git repo: hash the app/ package source as a fallback identifier.
    digest = hashlib.sha256()
    app_dir = os.path.join(PROJECT_ROOT, "app")
    for root, _dirs, files in os.walk(app_dir):
        for filename in sorted(files):
            if filename.endswith(".py"):
                digest.update(_sha256_file(os.path.join(root, filename)).encode("utf-8"))
    return {"type": "source_sha256", "value": digest.hexdigest(), "dirty": None}


def category_dir(category):
    return os.path.join(MODELS_ROOT, category)


def is_trained(category):
    directory = category_dir(category)
    return os.path.isfile(os.path.join(directory, MANIFEST_FILENAME)) and all(
        os.path.isfile(os.path.join(directory, filename)) for filename in ARTIFACT_FILENAMES.values()
    )


def list_trained_categories():
    if not os.path.isdir(MODELS_ROOT):
        return []
    return sorted(name for name in os.listdir(MODELS_ROOT) if is_trained(name))


def save(category, model, scaler, feature_columns, training_stats, dataset_snapshot, hyperparameters):
    """Writes model/scaler/features/training_stats + a signed manifest for `category`.

    dataset_snapshot: list of {"file", "sha256", "rows", "date_min", "date_max"} dicts,
      one per training CSV actually used (see app/trainer.py).
    hyperparameters: dict of everything needed to reproduce the run (RF/XGB params,
      effective_swing_threshold, lookforward_periods, min_hold_periods, scale_pos_weight).
    """
    final_dir = category_dir(category)
    os.makedirs(MODELS_ROOT, exist_ok=True)

    staging_dir = tempfile.mkdtemp(dir=MODELS_ROOT)
    try:
        artifacts = {"model": model, "scaler": scaler, "features": feature_columns, "training_stats": training_stats}
        artifact_hashes = {}
        for key, filename in ARTIFACT_FILENAMES.items():
            path = os.path.join(staging_dir, filename)
            joblib.dump(artifacts[key], path)
            artifact_hashes[filename] = _sha256_file(path)

        manifest = {
            "category": category,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "library_versions": _library_versions(),
            "code_version": _code_version(),
            "hyperparameters": hyperparameters,
            "dataset_snapshot": dataset_snapshot,
            "artifact_sha256": artifact_hashes,
        }
        signature = _sign_manifest(manifest)

        with open(os.path.join(staging_dir, MANIFEST_FILENAME), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        with open(os.path.join(staging_dir, SIGNATURE_FILENAME), "w", encoding="utf-8") as handle:
            handle.write(signature)

        if os.path.isdir(final_dir):
            shutil.rmtree(final_dir)
        shutil.move(staging_dir, final_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return manifest


def update_decision_threshold(category, new_threshold):
    """Changes a trained category's decision_threshold without retraining -- the RF/XGB
    trees don't depend on it (it only decides how a predicted probability becomes a trade
    decision, see app/detector.py's walk_forward_backtest), so refitting identical trees
    just to change this would be wasted compute. Re-saves through the normal signed path
    so the manifest/signature/hashes stay consistent -- hand-editing training_stats.pkl on
    disk would just make load() correctly refuse it as tampered.
    """
    loaded = load(category)  # raises ModelIntegrityError/ModelNotFoundError same as any load

    training_stats = dict(loaded.training_stats)
    training_stats["decision_threshold"] = new_threshold

    model = loaded.model
    if hasattr(model, "decision_threshold"):
        model.decision_threshold = new_threshold

    hyperparameters = dict(loaded.manifest.get("hyperparameters", {}))
    hyperparameters["decision_threshold"] = new_threshold

    return save(
        category,
        model=model,
        scaler=loaded.scaler,
        feature_columns=loaded.feature_columns,
        training_stats=training_stats,
        dataset_snapshot=loaded.manifest.get("dataset_snapshot", []),
        hyperparameters=hyperparameters,
    )


class LoadedModel:
    def __init__(self, model, scaler, feature_columns, training_stats, manifest, version_warnings):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.training_stats = training_stats
        self.manifest = manifest
        self.version_warnings = version_warnings


def load(category):
    directory = category_dir(category)
    manifest_path = os.path.join(directory, MANIFEST_FILENAME)
    signature_path = os.path.join(directory, SIGNATURE_FILENAME)

    if not os.path.isfile(manifest_path) or not os.path.isfile(signature_path):
        raise ModelNotFoundError(f"No trained model found for category '{category}' in {directory}")

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(signature_path, "r", encoding="utf-8") as handle:
        recorded_signature = handle.read().strip()

    expected_signature = _sign_manifest(manifest)
    if not hmac.compare_digest(expected_signature, recorded_signature):
        raise ModelIntegrityError(
            f"Manifest signature mismatch for '{category}' -- the manifest was modified or "
            "MODEL_SIGNING_KEY has changed since this model was saved. Refusing to load."
        )

    for filename, recorded_hash in manifest.get("artifact_sha256", {}).items():
        artifact_path = os.path.join(directory, filename)
        if not os.path.isfile(artifact_path):
            raise ModelIntegrityError(f"Missing artifact '{filename}' for category '{category}'.")
        actual_hash = _sha256_file(artifact_path)
        if not hmac.compare_digest(actual_hash, recorded_hash):
            raise ModelIntegrityError(
                f"Artifact '{filename}' for category '{category}' does not match the signed manifest "
                f"(expected sha256 {recorded_hash[:12]}..., got {actual_hash[:12]}...). Refusing to load."
            )

    version_warnings = []
    current_versions = _library_versions()
    for library, recorded_version in manifest.get("library_versions", {}).items():
        current_version = current_versions.get(library)
        if recorded_version and current_version and recorded_version != current_version:
            warning = f"{library}: trained with {recorded_version}, running {current_version}"
            version_warnings.append(warning)
            print(f"WARNING: model/library version mismatch for '{category}' -- {warning}")

    model = joblib.load(os.path.join(directory, ARTIFACT_FILENAMES["model"]))
    scaler = joblib.load(os.path.join(directory, ARTIFACT_FILENAMES["scaler"]))
    feature_columns = joblib.load(os.path.join(directory, ARTIFACT_FILENAMES["features"]))
    training_stats = joblib.load(os.path.join(directory, ARTIFACT_FILENAMES["training_stats"]))

    return LoadedModel(model, scaler, feature_columns, training_stats, manifest, version_warnings)
