import os
import tempfile
from PIL import Image

import pytest

from backend import models


def test_load_models_and_available():
    # Attempt to load models and ensure available_models returns a list
    models.load_models()
    av = models.available_models()
    assert isinstance(av, list)
    assert len(av) >= 1


def test_predict_heuristic():
    # Create a simple white image (likely classified as 'healthy' by heuristic)
    fd, path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    img = Image.new('RGB', (224, 224), color=(255, 255, 255))
    img.save(path)

    res = models.predict_image_sync(path, 'heuristic')
    assert isinstance(res, dict)
    assert 'label' in res and 'confidence' in res and 'recommendation' in res

    os.remove(path)


def test_predict_mango_v4_if_available():
    av = models.available_models()
    if 'mango_v4' not in av:
        pytest.skip("mango_v4 not available on this environment")

    fd, path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    img = Image.new('RGB', (224, 224), color=(255, 255, 255))
    img.save(path)

    res = models.predict_image_sync(path, 'mango_v4')
    assert isinstance(res, dict)
    assert 'label' in res
    assert 'recommendation' in res
    # recommendation should have stage or diagnosis keys
    rec = res['recommendation']
    assert isinstance(rec, dict)
    assert 'stage' in rec and 'treatment' in rec

    os.remove(path)


def test_fallback_when_model_not_loaded():
    # Simulate model file present but model not loaded by clearing _models
    # and ensuring model file exists; predict should return a fallback dict with fallback=True
    if not os.path.exists(models.MODEL_B_PATH):
        pytest.skip("No mango_model_v4_plus.h5 file present to test fallback")

    # Clear any loaded models to force fallback path
    models._models.clear()

    fd, path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    img = Image.new('RGB', (224, 224), color=(255, 255, 255))
    img.save(path)

    res = models.predict_image_sync(path, 'mango_v4')
    assert isinstance(res, dict)
    # If TensorFlow is present we expect the backend to mark fallback=True when model file
    # exists but model wasn't loaded. If TF is not present, the heuristic is used and
    # we still accept that as valid behaviour.
    if models._HAS_TF:
        assert res.get('fallback', False) is True
    assert 'label' in res

    os.remove(path)


def test_heuristic_recommendation_levels():
    # Heurística debe devolver un label como 'nivel_X' y una recomendación con claves esperadas
    fd, path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    img = Image.new('RGB', (224, 224), color=(230, 230, 230))
    img.save(path)

    res = models.predict_image_sync(path, 'heuristic')
    assert isinstance(res, dict)
    assert 'label' in res and isinstance(res['label'], str)
    assert res['label'].startswith('nivel_')

    rec = res.get('recommendation')
    assert isinstance(rec, dict)
    # Esperamos claves de stage y tratamiento (treatment) y lista de medicamentos
    assert 'stage' in rec and 'treatment' in rec and 'medications' in rec

    os.remove(path)
