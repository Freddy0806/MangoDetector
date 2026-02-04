import os
import json
from PIL import Image, ImageStat
# Support both package and direct script execution
try:
    from .recommendations import get_recommendation
except Exception:
    try:
        # Try top-level import if module executed directly
        from recommendations import get_recommendation
    except Exception:
        # Fallback: try absolute import via package name
        import importlib
        try:
            _pkg = importlib.import_module('backend.recommendations')
            get_recommendation = _pkg.get_recommendation
        except Exception:
            raise

import asyncio
import logging

logger = logging.getLogger(__name__)

# Try to import TensorFlow; if not available, use a heuristic fallback for offline inference
_HAS_TF = True
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import numpy as np
    Interpreter = None
except Exception:
    _HAS_TF = False
    load_model = None
    image = None
    np = None
    Interpreter = None

# Try tflite_runtime or ai-edge-litert if TF not available
if not _HAS_TF:
    # Try modern ai-edge-litert first
    try:
        from ai_edge_litert.interpreter import Interpreter
        import numpy as np
    except ImportError:
        # Fallback to legacy tflite-runtime
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
            import numpy as np
        except ImportError:
            pass


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_A_PATH = os.path.join(BASE_DIR, "keras_model.h5")
MODEL_B_PATH = os.path.join(BASE_DIR, "mango_model_v4_plus.h5")
MODEL_C_PATH = os.path.join(BASE_DIR, "keras_model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.txt")

_models = {}
_labels = []
_class_indices = {}
_load_errors = {}  # model_name -> error message (if loading failed)


def _load_labels():
    global _labels
    if os.path.exists(LABELS_PATH):
        labels = []
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                # If the line starts with an index, e.g. '0 Nivel 0', use the remainder as label
                try:
                    int(parts[0])
                    label = ' '.join(parts[1:]).strip()
                except Exception:
                    label = s
                labels.append(label)
        _labels = labels
    else:
        _labels = []


def _load_class_indices():
    global _class_indices
    if os.path.exists(CLASS_INDICES_PATH):
        try:
            with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
                # Try JSON first
                try:
                    parsed = json.loads(content)
                except Exception:
                    parsed = None
                if not parsed:
                    # Try to interpret a Python dict literal (some files use single quotes)
                    try:
                        import ast

                        parsed = ast.literal_eval(content)
                    except Exception:
                        parsed = None
                if parsed and isinstance(parsed, dict):
                    # If parsed is a mapping label->int, invert to int->label
                    keys_are_str_vals_are_ints = all(isinstance(k, str) and isinstance(v, int) for k, v in parsed.items())
                    if keys_are_str_vals_are_ints:
                        inv = {v: k for k, v in parsed.items()}
                        _class_indices = inv
                    else:
                        # assume it's already int->label or similar
                        _class_indices = parsed
                else:
                    # Fallback: Try line based mapping "0 label"
                    mapping = {}
                    for line in content.splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            key = parts[0]
                            value = " ".join(parts[1:])
                            try:
                                mapping[int(key)] = value
                            except Exception:
                                # fallback: mapping strings to ints
                                mapping[key] = value
                    _class_indices = mapping
        except Exception:
            _class_indices = {}
    else:
        _class_indices = {}


def load_models():
    """Cargar modelos en memoria para uso síncrono (modo local). Si TensorFlow no está instalado,
    cargamos etiquetas y usamos un predictor heurístico basado en imagen.
    """
    global _models
    _load_labels()
    _load_class_indices()
    if not _HAS_TF:
        # No TF: skip loading heavy model files
        logger.warning("TensorFlow no disponible. Usando predicción heurística offline.")
        _models = {}
        return
    # Try to load models that are present
    loaded = {}
    def _build_compat_shims():
        """Return a dict of compatibility shim classes for known layer compatibility issues.

        For example, some models saved with newer Keras versions include kwargs that older
        deserializers don't accept (e.g., 'groups' passed to DepthwiseConv2D). We provide a
        wrapper that silently drops unsupported kwargs during deserialization.
        """
        shims = {}
        try:
            from tensorflow.keras.layers import DepthwiseConv2D as TKDepthwiseConv2D

            class DepthwiseConv2DShim(TKDepthwiseConv2D):
                def __init__(self, *args, **kwargs):
                    # Drop unknown compatibility kwargs commonly present in saved configs
                    kwargs.pop('groups', None)
                    kwargs.pop('filters', None)  # in case a saved config added this erroneously
                    super().__init__(*args, **kwargs)

            shims['DepthwiseConv2D'] = DepthwiseConv2DShim
        except Exception:
            # If import fails, don't add shim
            pass
        return shims

    compat = _build_compat_shims()

    def _validate_model(name, model):
        """Run a quick inference with a synthetic input to ensure the model works end-to-end.

        Supports single and multi-input models by preparing inputs accordingly.
        """
        try:
            import numpy as _np
            # prepare dummy input(s) using helper
            shapes = _get_input_shapes(model)
            dummy_inputs = []
            for (w, h) in shapes:
                dummy_inputs.append(_np.zeros((1, h, w, 3), dtype='float32'))
            inp = dummy_inputs[0] if len(dummy_inputs) == 1 else dummy_inputs
            model.predict(inp)
            return True, None
        except Exception as ex:
            return False, str(ex)

    if os.path.exists(MODEL_A_PATH):
        try:
            # load_model with compile=False avoids issues with missing optimizer or custom training objects
            _models['labels_model'] = load_model(MODEL_A_PATH, compile=False)
            # validate model by running a small dummy predict
            ok, err = _validate_model('labels_model', _models['labels_model'])
            if not ok:
                _load_errors['labels_model'] = f"Runtime inference error: {err}"
                logger.warning("Modelo %s cargado pero falló durante inferencia de prueba: %s", MODEL_A_PATH, err)
                del _models['labels_model']
            else:
                loaded['labels_model'] = MODEL_A_PATH
                logger.info("Cargado y validado: %s", MODEL_A_PATH)
        except TypeError as e:
            # Try again with compatibility shims
            logger.warning("TypeError cargando %s: %s. Intentando carga con shims de compatibilidad.", MODEL_A_PATH, e)
            try:
                _models['labels_model'] = load_model(MODEL_A_PATH, custom_objects=compat, compile=False)
                ok, err = _validate_model('labels_model', _models['labels_model'])
                if not ok:
                    _load_errors['labels_model'] = f"Runtime inference error after shims: {err}"
                    logger.warning("Modelo %s falló durante inferencia tras aplicar shims: %s", MODEL_A_PATH, err)
                    del _models['labels_model']
                else:
                    loaded['labels_model'] = MODEL_A_PATH
                    logger.info("Cargado con shims y validado: %s", MODEL_A_PATH)
            except Exception as ex:
                _load_errors['labels_model'] = str(ex)
                logger.exception("Error cargando %s incluso con shims", MODEL_A_PATH)
        except Exception as e:
            _load_errors['labels_model'] = str(e)
            logger.exception("Error cargando %s", MODEL_A_PATH)
    if os.path.exists(MODEL_B_PATH):
        try:
            _models['mango_v4'] = load_model(MODEL_B_PATH, compile=False)
            ok, err = _validate_model('mango_v4', _models['mango_v4'])
            if not ok:
                _load_errors['mango_v4'] = f"Runtime inference error: {err}"
                logger.warning("Modelo %s cargado pero falló durante inferencia de prueba: %s", MODEL_B_PATH, err)
                del _models['mango_v4']
            else:
                loaded['mango_v4'] = MODEL_B_PATH
                logger.info("Cargado y validado: %s", MODEL_B_PATH)
        except TypeError as e:
            logger.warning("TypeError cargando %s: %s. Intentando carga con shims de compatibilidad.", MODEL_B_PATH, e)
            try:
                _models['mango_v4'] = load_model(MODEL_B_PATH, custom_objects=compat, compile=False)
                ok, err = _validate_model('mango_v4', _models['mango_v4'])
                if not ok:
                    _load_errors['mango_v4'] = f"Runtime inference error after shims: {err}"
                    logger.warning("Modelo %s falló durante inferencia tras aplicar shims: %s", MODEL_B_PATH, err)
                    del _models['mango_v4']
                else:
                    loaded['mango_v4'] = MODEL_B_PATH
                    logger.info("Cargado con shims y validado: %s", MODEL_B_PATH)
            except Exception as ex:
                _load_errors['mango_v4'] = str(ex)
                logger.exception("Error cargando %s incluso con shims", MODEL_B_PATH)
        except Exception as e:
            _load_errors['mango_v4'] = str(e)
            logger.exception("Error cargando %s", MODEL_B_PATH)
            # Provide a clearer runtime hint for common HDF5/compatibility problems
            if 'HDF5' in str(e) or 'h5py' in str(e) or 'not a valid' in str(e):
                logger.error("Comprueba que 'h5py' esté instalado y que '%s' no esté corrupto.", MODEL_B_PATH)

    if os.path.exists(MODEL_C_PATH):
        try:
            if _HAS_TF:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=MODEL_C_PATH)
            elif Interpreter is not None:
                interpreter = Interpreter(model_path=MODEL_C_PATH)
            else:
                interpreter = None
            
            if interpreter:
                interpreter.allocate_tensors()
                _models['keras_tflite'] = interpreter
                logger.info("Cargado TFLite: %s", MODEL_C_PATH)
            else:
                logger.warning("No se pudo cargar TFLite: ni TensorFlow ni tflite_runtime disponibles.")
        except Exception as e:
            _load_errors['keras_tflite'] = str(e)
            logger.exception("Error cargando TFLite %s", MODEL_C_PATH)

    if loaded:
        logger.info("Modelos cargados: %s", loaded)
    else:
        logger.info("No se cargaron modelos Keras. Usando heurística si es necesario.")


def _predict_tflite(interpreter, path: str):
    """Realiza inferencia usando un intérprete TFLite."""
    try:
        import numpy as _np
    except ImportError:
        raise RuntimeError("Numpy no disponible para TFLite")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input shape
    # TFLite input shape usually [1, h, w, 3]
    input_shape = input_details[0]['shape']
    h, w = input_shape[1], input_shape[2]
    
    # Load and resize image
    img = Image.open(path).convert('RGB').resize((w, h), Image.LANCZOS)
    
    # Check input type (float32 vs uint8) and normalize if needed
    input_data = _np.expand_dims(img, axis=0)
    if input_details[0]['dtype'] == _np.float32:
        input_data = (np.float32(input_data) / 255.0)
    else:
        # uint8 models might expect 0-255
        input_data = _np.uint8(input_data)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Process output similar to Keras models
    preds_arr = _np.array(output_data)
    if preds_arr.ndim == 2 and preds_arr.shape[1] > 1:
        idx = int(_np.argmax(preds_arr[0]))
        conf = float(_np.max(preds_arr[0]))
    elif preds_arr.ndim == 1:
        idx = int(_np.argmax(preds_arr))
        conf = float(_np.max(preds_arr))
    else:
        flat = preds_arr.flatten()
        idx = int(_np.argmax(flat))
        conf = float(_np.max(flat))

    return idx, conf


def _heuristic_predict(path: str):
    """Heurística mejorada:
    - Convierte a escala de grises y calcula porcentaje de píxeles oscuros (lesiones).
    - Mapea el resultado a un `nivel_0 .. nivel_4` y devuelve recomendación y tratamiento consistente
      con lo que devuelve el modelo entrenado (para mantener la UX homogénea).
    """
    img = Image.open(path).convert('L')
    stat = ImageStat.Stat(img)
    mean = stat.mean[0]

    # Calculate dark pixel ratio (simple threshold)
    bw = img.point(lambda p: 1 if p < 80 else 0)
    dark_pixels = sum(bw.getdata())
    total = img.size[0] * img.size[1]
    ratio = dark_pixels / float(total)

    # Map ratio -> nivel (0..4) and confidence
    if ratio < 0.01 and mean > 200:
        level = 0
        conf = 0.95
    elif ratio < 0.03:
        level = 1
        conf = min(0.6 + (0.03 - ratio) * 8, 0.85)
    elif ratio < 0.08:
        level = 2
        conf = min(0.6 + (ratio - 0.03) * 6, 0.9)
    elif ratio < 0.15:
        level = 3
        conf = min(0.65 + (ratio - 0.08) * 4, 0.95)
    else:
        level = 4
        conf = min(0.75 + (ratio - 0.15) * 2, 0.99)

    label = f'nivel_{level}'
    recommendation = get_recommendation(label)
    return {"label": label, "confidence": float(conf), "recommendation": recommendation}

def predict_image_sync(path: str, model_name: str):
    """Versión síncrona de predicción para uso desde la UI local.
    Si TensorFlow está disponible y el modelo fue cargado, se usa la inferencia Keras; de lo contrario,
    se usa una heurística basada en la imagen para detectar manchas oscuras.
    Devuelve diccionario {label, confidence, recommendation}.
    """
    # Reload models on demand if TF is available but models were not loaded
    if _HAS_TF and not _models:
        try:
            load_models()
        except Exception:
            pass

    # If TensorFlow is not available and the caller requested a specific model, use heuristic
    if not _HAS_TF and model_name != 'heuristic':
        logger.warning("TensorFlow no disponible; usando heurística como fallback para '%s'", model_name)
        h = _heuristic_predict(path)
        h['fallback'] = True
        h['fallback_source'] = 'no_tensorflow'
        return h

    if _HAS_TF and model_name in _models:
        model = _models[model_name]
        
        # Dispatch for TFLite
        if model_name == 'keras_tflite':
            try:
                idx, conf = _predict_tflite(model, path)
            except Exception as ex:
                logger.exception('Error inferencia TFLite %s: %s', model_name, ex)
                raise RuntimeError(f"Error TFLite: {ex}")
            
            # Map index to label (using labels.txt logic often associated with this converted model)
            if _labels:
                label = _labels[idx] if idx < len(_labels) else str(idx)
            else:
                label = str(idx)
            
            recommendation = get_recommendation(label)
            return {"label": label, "confidence": conf, "recommendation": recommendation}

        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"No se pudo abrir la imagen: {e}")

        # Prepare inputs respecting possible multiple input signatures
        inputs = _prepare_inputs_for_model(model, path)

        try:
            preds = model.predict(inputs)
        except Exception as ex:
            logger.exception('Error durante la inferencia con %s: %s', model_name, ex)
            raise RuntimeError(f"Error durante la inferencia del modelo '{model_name}': {ex}")

        # Normalize various possible output shapes
        try:
            preds_arr = np.array(preds)
            # If model outputs a batch x classes probabilities
            if preds_arr.ndim == 2 and preds_arr.shape[1] > 1:
                idx = int(np.argmax(preds_arr[0]))
                conf = float(np.max(preds_arr[0]))
            # If output is single probability per class (binary)
            elif preds_arr.ndim == 2 and preds_arr.shape[1] == 1:
                idx = int(preds_arr[0][0] > 0.5)
                conf = float(preds_arr[0][0])
            # If output is flat vector
            elif preds_arr.ndim == 1:
                idx = int(np.argmax(preds_arr))
                conf = float(np.max(preds_arr))
            else:
                # Fallback: try to flatten and pick highest
                flat = preds_arr.flatten()
                idx = int(np.argmax(flat))
                conf = float(np.max(flat))
        except Exception as ex:
            logger.exception('No se pudo interpretar la salida del modelo: %s', ex)
            raise RuntimeError('Salida del modelo en formato inesperado')

            # Map to labels
        if model_name == 'labels_model' and _labels:
            label = _labels[idx] if idx < len(_labels) else str(idx)
        elif model_name == 'mango_v4' and _class_indices:
            # _class_indices may map in various formats; ensure we get a human-friendly label
            # If mapping is like {"nivel_0": 0} invert it to {0: "nivel_0"}
            if isinstance(list(_class_indices.keys())[0], str):
                inv = {v: k for k, v in _class_indices.items()}
                label = inv.get(idx, str(idx))
            else:
                label = _class_indices.get(idx, str(idx))
        else:
            label = str(idx)

        recommendation = get_recommendation(label)

        return {"label": label, "confidence": conf, "recommendation": recommendation}

    # If requested model exists but TF not available or model not loaded, fallback gracefully using heuristic
    if model_name != 'heuristic' and _HAS_TF and model_name not in _models:
        # If model file exists, provide a mapped heuristic prediction so UI can continue operating
        file_exists = False
        if model_name == 'mango_v4':
            file_exists = os.path.exists(MODEL_B_PATH)
        elif model_name == 'labels_model':
            file_exists = os.path.exists(MODEL_A_PATH)
        elif model_name == 'keras_tflite':
             file_exists = os.path.exists(MODEL_C_PATH)

        if file_exists:
            logger.warning("Modelo '%s' no cargado, usando heurística mapeada como fallback.", model_name)
            fallback = _heuristic_predict(path)
            # Try to map heuristic label to available labels or class indices when possible
            mapped_label = fallback['label']
            if model_name == 'labels_model' and _labels:
                # Find best matching label by token overlap
                best = None
                best_score = 0
                q = mapped_label.lower()
                for i, lab in enumerate(_labels):
                    s = lab.lower()
                    score = sum(1 for t in q.split() if t in s)
                    if score > best_score:
                        best_score = score
                        best = lab
                if best:
                    mapped_label = best
            elif model_name == 'mango_v4' and _class_indices:
                # Try to find an index whose label contains heuristic label as substring
                q = mapped_label.lower()
                rev = {v.lower(): k for k, v in _class_indices.items()}
                # try exact substring
                found = None
                for k, v in _class_indices.items():
                    if q in v.lower() or v.lower() in q:
                        found = v
                        break
                if found:
                    mapped_label = found
            result = {
                'label': mapped_label,
                'confidence': fallback.get('confidence', 0.0),
                'recommendation': fallback.get('recommendation', {}),
                'fallback': True,
                'fallback_source': 'heuristic'
            }
            return result
        # Otherwise, raise informative error
        raise RuntimeError(f"Modelo '{model_name}' no cargado. Asegúrate de tener instalado TensorFlow y de que el archivo de modelo exista.")

    # fallback heuristic
    return _heuristic_predict(path)

def _get_input_shapes(model):
    """Return a list of (w,h) tuples for each model input shape.

    Supports models with single or multiple inputs.
    If shapes cannot be determined, returns [(224,224)].
    """
    try:
        shape = getattr(model, 'input_shape', None)
        if shape is None:
            try:
                shape = model.layers[0].input_shape
            except Exception:
                shape = None
        shapes = []
        if isinstance(shape, list):
            base_list = shape
        else:
            base_list = [shape]
        for s in base_list:
            if s and isinstance(s, (list, tuple)) and len(s) >= 3:
                h, w = int(s[1]) if s[1] else 224, int(s[2]) if s[2] else 224
                shapes.append((w, h))
        if not shapes:
            shapes = [(224, 224)]
        return shapes
    except Exception:
        return [(224, 224)]


def _prepare_inputs_for_model(model, path: str):
    """Prepare numpy array input(s) for a given model and image path.

    Returns either a single numpy array or a list of arrays corresponding to model inputs.
    """
    try:
        import numpy as _np
    except Exception:
        raise RuntimeError('numpy no disponible')

    shapes = _get_input_shapes(model)
    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir la imagen: {e}")

    arrays = []
    for (w, h) in shapes:
        im = img.resize((w, h), resample=Image.LANCZOS)
        arr = _np.array(im).astype('float32') / 255.0
        if arr.ndim == 3:
            arr = _np.expand_dims(arr, axis=0)
        arrays.append(arr)

    if len(arrays) == 1:
        return arrays[0]
    return arrays


async def predict_image(path: str, model_name: str):
    # Async wrapper: call sync impl in thread if necessary
    return predict_image_sync(path, model_name)


def get_labels():
    return _labels


def get_class_indices():
    return _class_indices


def get_model_load_errors():
    """Return a dict mapping model names to load error messages (if any)."""
    return dict(_load_errors)


def available_models():
    """Return a list of identifiers for models considered available for inference.

    - If TensorFlow is installed and models were successfully loaded, returns those keys.
    - Otherwise returns ['heuristic'] as the safe fallback.
    """
    # Ensure models are loaded if TF is available
    if _HAS_TF and not _models:
        try:
            load_models()
        except Exception:
            pass
    if _HAS_TF and _models:
        return list(_models.keys())
    return ["heuristic"]


def model_file_status():
    """Return a mapping of the model file existence state e.g. {'mango_v4': True}.

    Useful to let the UI show available files even when loading fails (and present
    helpful messages to the user).
    """
    return {
        'mango_v4': os.path.exists(MODEL_B_PATH),
        'labels_model': os.path.exists(MODEL_A_PATH),
        'keras_tflite': os.path.exists(MODEL_C_PATH),
    }


def test_model_inference(model_name: str = 'mango_v4'):
    """Run a quick inference with a synthetic image to validate model loading and output parsing.

    Returns the prediction dict or raises a RuntimeError with details.
    """
    # Create a white image
    fd, tmp = None, None
    from PIL import Image
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix='.jpg')
    try:
        os.close(fd)
        img = Image.new('RGB', (256, 256), (255, 255, 255))
        img.save(tmp)
        return predict_image_sync(tmp, model_name)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


if __name__ == '__main__':
    # Quick diagnostic when running this module directly
    logging.basicConfig(level=logging.INFO)
    logger.info('TensorFlow disponible: %s', _HAS_TF)
    load_models()
    logger.info('Modelos detectados: %s', available_models())
    logger.info('Etiquetas (labels.txt) cargadas: %d', len(get_labels()) if get_labels() else 0)
    logger.info('Class indices (class_indices.txt) cargados: %d', len(get_class_indices()) if get_class_indices() else 0)
    # If mango_v4 present, do a quick test inference and print result
    if 'mango_v4' in available_models():
        try:
            res = test_model_inference('mango_v4')
            logger.info('Prueba de inferencia (mango_v4): %s', res)
        except Exception as e:
            logger.exception('Error en prueba de inferencia: %s', e)
