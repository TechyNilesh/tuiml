"""Tests for tuiml.serving.schemas (Pydantic request/response models)."""

import pytest

from tuiml.serving.schemas import (
    PredictRequest,
    PredictResponse,
    PredictProbaResponse,
    LoadModelRequest,
    ModelInfoResponse,
    ModelListResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
    BatchPredictRequest,
    BatchPredictResponse,
)


# ---------------------------------------------------------------------------
# PredictRequest
# ---------------------------------------------------------------------------
class TestPredictRequest:
    def test_valid_construction(self):
        req = PredictRequest(features=[[1, 2], [3, 4]])
        assert req.features == [[1, 2], [3, 4]]

    def test_single_sample(self):
        req = PredictRequest(features=[[5.1, 3.5, 1.4, 0.2]])
        assert len(req.features) == 1

    def test_float_values(self):
        req = PredictRequest(features=[[1.5, 2.7], [3.1, 4.9]])
        assert req.features[0][0] == 1.5

    def test_missing_features_raises(self):
        with pytest.raises(Exception):
            PredictRequest()


# ---------------------------------------------------------------------------
# PredictResponse
# ---------------------------------------------------------------------------
class TestPredictResponse:
    def test_valid_construction(self):
        resp = PredictResponse(predictions=[0, 1, 0], model_id="clf_1")
        assert resp.predictions == [0, 1, 0]
        assert resp.model_id == "clf_1"
        assert resp.model_class is None

    def test_with_model_class(self):
        resp = PredictResponse(
            predictions=[1, 0],
            model_id="test",
            model_class="NaiveBayes",
        )
        assert resp.model_class == "NaiveBayes"


# ---------------------------------------------------------------------------
# PredictProbaResponse
# ---------------------------------------------------------------------------
class TestPredictProbaResponse:
    def test_valid_construction(self):
        resp = PredictProbaResponse(
            probabilities=[[0.8, 0.2], [0.3, 0.7]],
            model_id="clf_1",
        )
        assert len(resp.probabilities) == 2
        assert resp.classes is None

    def test_with_classes(self):
        resp = PredictProbaResponse(
            probabilities=[[0.9, 0.1]],
            classes=["cat", "dog"],
            model_id="clf",
        )
        assert resp.classes == ["cat", "dog"]


# ---------------------------------------------------------------------------
# LoadModelRequest
# ---------------------------------------------------------------------------
class TestLoadModelRequest:
    def test_valid_construction(self):
        req = LoadModelRequest(model_id="my_model", path="models/clf.pkl")
        assert req.model_id == "my_model"
        assert req.path == "models/clf.pkl"
        assert req.metadata is None

    def test_with_metadata(self):
        req = LoadModelRequest(
            model_id="m1",
            path="/tmp/model.pkl",
            metadata={"version": "1.0", "dataset": "iris"},
        )
        assert req.metadata["version"] == "1.0"


# ---------------------------------------------------------------------------
# ModelInfoResponse
# ---------------------------------------------------------------------------
class TestModelInfoResponse:
    def test_valid_construction(self):
        resp = ModelInfoResponse(
            model_id="m1",
            model_class="RandomForest",
            model_module="tuiml.algorithms.ensemble",
            params={"n_trees": 100},
            path="/models/rf.pkl",
            loaded_at="2025-01-01T00:00:00",
            prediction_count=42,
            metadata={"tag": "production"},
        )
        assert resp.model_id == "m1"
        assert resp.model_class == "RandomForest"
        assert resp.prediction_count == 42

    def test_defaults(self):
        resp = ModelInfoResponse(model_id="m1")
        assert resp.model_class is None
        assert resp.model_module is None
        assert resp.params == {}
        assert resp.prediction_count == 0
        assert resp.metadata == {}


# ---------------------------------------------------------------------------
# ModelListResponse
# ---------------------------------------------------------------------------
class TestModelListResponse:
    def test_valid_construction(self):
        resp = ModelListResponse(models=["m1", "m2", "m3"], count=3)
        assert resp.count == 3
        assert "m1" in resp.models

    def test_empty_list(self):
        resp = ModelListResponse(models=[], count=0)
        assert resp.count == 0
        assert resp.models == []


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------
class TestHealthResponse:
    def test_valid_construction(self):
        resp = HealthResponse(status="healthy", version="0.1.1", models_loaded=2)
        assert resp.status == "healthy"
        assert resp.version == "0.1.1"
        assert resp.models_loaded == 2


# ---------------------------------------------------------------------------
# StatsResponse
# ---------------------------------------------------------------------------
class TestStatsResponse:
    def test_valid_construction(self):
        resp = StatsResponse(
            loaded_models=2,
            max_models=10,
            total_predictions=500,
            models=[
                {"model_id": "m1", "model_class": "SVM", "prediction_count": 300},
                {"model_id": "m2", "model_class": "NB", "prediction_count": 200},
            ],
        )
        assert resp.loaded_models == 2
        assert resp.total_predictions == 500
        assert len(resp.models) == 2


# ---------------------------------------------------------------------------
# ErrorResponse
# ---------------------------------------------------------------------------
class TestErrorResponse:
    def test_valid_construction(self):
        resp = ErrorResponse(error="Model not found")
        assert resp.error == "Model not found"
        assert resp.detail is None
        assert resp.model_id is None

    def test_with_all_fields(self):
        resp = ErrorResponse(
            error="Prediction failed",
            detail="Input shape mismatch",
            model_id="broken_model",
        )
        assert resp.detail == "Input shape mismatch"
        assert resp.model_id == "broken_model"


# ---------------------------------------------------------------------------
# BatchPredictRequest
# ---------------------------------------------------------------------------
class TestBatchPredictRequest:
    def test_valid_construction(self):
        req = BatchPredictRequest(
            requests=[
                {"model_id": "m1", "features": [[1, 2]]},
                {"model_id": "m2", "features": [[3, 4]]},
            ]
        )
        assert len(req.requests) == 2


# ---------------------------------------------------------------------------
# BatchPredictResponse
# ---------------------------------------------------------------------------
class TestBatchPredictResponse:
    def test_valid_construction(self):
        resp = BatchPredictResponse(
            results=[
                {"model_id": "m1", "predictions": [0]},
                {"model_id": "m2", "error": "not loaded"},
            ],
            successful=1,
            failed=1,
        )
        assert resp.successful == 1
        assert resp.failed == 1
        assert len(resp.results) == 2
