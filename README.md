# MangoGuard

Aplicación para detectar antracnosis en frutos de mango usando modelos de visión por computadora.

## Componentes

- Aplicación local: todo funciona sin servidor (modo offline) desde la app Flet (ideal para móvil y escritorio). 
- Frontend: Flet (único código Python que funciona en web, móvil y escritorio) — la UI carga los modelos y almacena usuarios/analisis en archivos locales.
- Almacenamiento local: datos son guardados en `data/` (JSON), y contraseñas se hashean con **bcrypt**.
- Modelos: `keras_model.h5` (usa `labels.txt`) y `mango_model_v4_plus.h5` (usa `class_indices.txt`)

## Configuración

1. Copiar `.env.example` a `.env` y ajustar `MONGO_URI` y `SECRET_KEY`.
2. Instalar dependencias: `pip install -r requirements.txt`.
3. Ejecutar el backend: `uvicorn backend.app:app --reload --port 8000`.
4. Ejecutar el frontend: `python frontend/app.py` (abrirá en el navegador o se puede empaquetar para móvil/escritorio con Flet).

## Notas

- Ajusta los mapeos de `backend/recommendations.py` para que coincidan con tus clases reales.
- Los modelos se cargan automáticamente al iniciar el backend y se buscan en el directorio raíz del proyecto.
