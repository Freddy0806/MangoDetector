"""Flet frontend for MangoGuard.

Contains the UI (login/register/dashboard) and interactions with local storage
and model inference helpers.
"""

from __future__ import annotations

import logging
import flet as ft
from flet import Colors, Icons
import os
import json
import sys
import tempfile
import contextlib
from threading import Thread
from uuid import uuid4

logger = logging.getLogger(__name__)  # module logger

# Optionally enable tracemalloc for detailed allocation traces when debugging memory warnings.
if os.getenv('MANGOGUARD_ENABLE_TRACEMALLOC', '').lower() in ('1', 'true', 'yes'):
    try:
        import tracemalloc

        tracemalloc.start()
        logger.info('tracemalloc started')
    except Exception as ex:  # pragma: no cover - optional
        logger.warning('Could not start tracemalloc: %s', ex)

# Ensure project root is on sys.path so backend package can be imported when running script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Local-only mode: import local storage and sync prediction
from backend.local_db import create_user, authenticate_user, add_analysis, get_user_history
from backend.models import load_models, predict_image_sync, get_labels, get_class_indices, available_models, model_file_status, get_model_load_errors

# --- UI palette (modern green + neutrals)
PALETTE = {
    'primary': '#1B5E20',      # deep green
    'primary_light': '#66BB6A',
    'accent': '#A5D6A7',
    'bg_light': '#F6FBF6',
    'card_light': '#FFFFFF',
    'bg_dark': '#0F1720',
    'card_dark': '#1E293B',
    'text_light': '#0F1720',
    'text_dark': '#F8FAFC',
    'muted_light': '#6B7280',
    'muted_dark': '#94A3B8'
}


# Auth persistence (store user minimal info locally)
AUTH_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'auth.json')

def save_current_user(user: dict):
    try:
        with open(AUTH_FILE, 'w', encoding='utf-8') as f:
            json.dump({'user': {'id': user.get('id'), 'username': user.get('username'), 'email': user.get('email')}}, f)
    except Exception:
        pass


def load_current_user():
    try:
        with open(AUTH_FILE, 'r', encoding='utf-8') as f:
            return json.load(f).get('user')
    except Exception:
        return None


class MangoApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "MangoGuard"
        # Theme: default to LIGHT (no system adaptation)
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.current_user = load_current_user()
        self.page.padding = 20

        # Minimal & modern inputs
        self.email = ft.TextField(label="Email", width=340, text_size=14)
        self.password = ft.TextField(label="Contraseña", password=True, can_reveal_password=True, width=340, text_size=14)
        self.username = ft.TextField(label="Usuario", width=340, text_size=14)
        
        # Absolute upload directory
        self.upload_dir = os.path.join(PROJECT_ROOT, "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)

        # Flet FilePicker for cross-platform (web/mobile/desktop) client-side file selection
        self.file_picker = ft.FilePicker(on_result=self.file_picker_result, on_upload=self.on_upload_progress)
        self.page.overlay.append(self.file_picker)
        self.file_picker_available = True
        logger.info("FilePicker enabled")

        # dynamic model selector (filled later)
        self.model_selector = ft.Dropdown()

        # set theme and colors
        self.is_dark = self.page.theme_mode == ft.ThemeMode.DARK
        self.update_theme()

        # Load models
        try:
            load_models()
        except Exception as ex:
            logger.warning("Fallo al cargar modelos: %s", ex, exc_info=True)

        # model selector will be populated when dashboard is built
        # UI helper: status text shown when opening file selector
        self.picker_status = ft.Text("", size=12, color=self.muted)

        # main view (ListView for scrolling to see all analyses)
        # Enable page-level automatic scrolling and make ListView expand to available space
        try:
            self.page.scroll = ft.ScrollMode.AUTO
        except Exception:
            # older flet versions may not have ScrollMode; ignore gracefully
            pass
        self.main_view = ft.ListView(spacing=18, padding=20, auto_scroll=False, expand=True)
        self.page.add(self.main_view)
        if self.current_user:
            self.build_dashboard()
        else:
            self.build_login()

    def update_model_selector(self):
        loaded = set(available_models())
        files = model_file_status()
        candidates = set(loaded)
        # also offer selection for model files even if they failed to load
        for k, exists in files.items():
            if exists:
                candidates.add(k)
        options = []
        for m in sorted(candidates):
            label = m
            suffix = ''
            if m == 'heuristic':
                label = 'Heurística (sin TensorFlow)'
            elif m == 'mango_v4':
                label = 'Mango model (v4)'
            elif m == 'labels_model':
                label = 'Labels model'
            elif m == 'keras_tflite':
                label = 'Modelo TFLite (keras_model)'
            if m not in loaded and m != 'heuristic':
                suffix = ' (no cargado)'
            options.append(ft.dropdown.Option(m, text=label + suffix))
        self.model_selector.options = options
        # Prefer mango_v4 as default when available
        if options:
            # find mango_v4 option if present
            mango_opt = next((o for o in options if o.key == 'mango_v4'), None)
            if mango_opt:
                self.model_selector.value = mango_opt.key
            else:
                self.model_selector.value = options[0].key
            try:
                self.model_selector.update()
            except Exception:
                # control may not yet be added to the page; update later when placed
                pass

    def update_theme(self):
        """Aplica la paleta según "is_dark" y guarda colores en self.* para usar en la UI."""
        if self.is_dark:
            self.page.bgcolor = PALETTE['bg_dark']
            self.card_bg = PALETTE['card_dark']
            self.text_color = PALETTE['text_dark']
            self.muted = PALETTE['muted_dark']
            self.page.theme_mode = ft.ThemeMode.DARK
            self.input_bg = '#0B1220'
        else:
            self.page.bgcolor = PALETTE['bg_light']
            self.card_bg = PALETTE['card_light']
            self.text_color = PALETTE['text_light']
            self.muted = PALETTE['muted_light']
            self.page.theme_mode = ft.ThemeMode.LIGHT
            self.input_bg = '#F4FBF4'
        self.primary = PALETTE['primary']
        self.accent = PALETTE['accent']
        # Update theme object for consistency (compatible con múltiples versiones de Flet)
        try:
            self.page.theme = ft.Theme(primary_color=self.primary)
        except TypeError:
            # Fallback: crear Theme sin argumentos y asignar atributo si existe
            try:
                t = ft.Theme()
                if hasattr(t, 'primary_color'):
                    setattr(t, 'primary_color', self.primary)
                elif hasattr(t, 'primary'):
                    setattr(t, 'primary', self.primary)
                self.page.theme = t
            except Exception:
                # última opción: confiar en colores explícitos definidos en la UI
                pass
        self.page.update()

    def toggle_theme(self, e=None):
        self.is_dark = not self.is_dark
        self.update_theme()
        # Rebuild current view to apply colors
        if self.current_user:
            self.build_dashboard()
        else:
            self.build_login()
        self.page.update()

    def build_login(self):
        self.main_view.controls.clear()
        # styled fields
        self.email.filled = True
        self.email.bgcolor = self.input_bg
        self.email.border = None
        self.email.prefix_icon = Icons.MAIL
        self.password.filled = True
        self.password.bgcolor = self.input_bg
        self.password.border = None
        self.password.prefix_icon = Icons.LOCK

        hero = ft.Row([ft.Text("🥭 MangoGuard", size=28, weight=ft.FontWeight.BOLD, color=self.primary), ft.VerticalDivider(width=10), ft.Text("Detección de antracnosis", size=14, color=self.muted)])

        card = ft.Card(content=ft.Container(content=ft.Column([
            hero,
            ft.Divider(height=8),
            self.email,
            self.password,
            ft.Row([
                ft.ElevatedButton("Ingresar", on_click=self.login, bgcolor=self.primary, color=Colors.WHITE, width=160),
                ft.TextButton("Registrarse", on_click=self.build_register)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Divider(height=6),
            ft.Text("Protegido y sin conexión — tus datos se guardan localmente.", size=12, color=self.muted)
        ], spacing=12), padding=22, width=520, bgcolor=self.card_bg, border_radius=14), elevation=4)

        alignment_center = getattr(ft.alignment, 'center', None)
        if alignment_center is None:
            alignment_center = getattr(ft.alignment, 'CENTER', None)
        if alignment_center is None:
            container = ft.Container(content=card)
        else:
            container = ft.Container(content=card, alignment=alignment_center)
        self.main_view.controls.append(container)
        self.page.update()

    def build_register(self, e=None):
        self.main_view.controls.clear()
        # styled fields
        self.username.filled = True
        self.username.bgcolor = self.input_bg
        self.username.border = None
        self.email.filled = True
        self.email.bgcolor = self.input_bg
        self.email.border = None
        self.password.filled = True
        self.password.bgcolor = self.input_bg
        self.password.border = None

        card = ft.Card(content=ft.Container(content=ft.Column([
            ft.Row([ft.Text("Crear cuenta", size=20, weight=ft.FontWeight.BOLD, color=self.primary), ft.IconButton(icon=Icons.DARK_MODE, on_click=self.toggle_theme, tooltip='Alternar tema')]),
            self.username,
            self.email,
            self.password,
            ft.Row([
                ft.ElevatedButton("Crear", on_click=self.register, bgcolor=self.primary, color=Colors.WHITE, width=160),
                ft.TextButton("Volver", on_click=lambda e: self.build_login())
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        ], spacing=12), padding=22, width=520, bgcolor=self.card_bg, border_radius=14), elevation=4)

        alignment_center = getattr(ft.alignment, 'center', None)
        if alignment_center is None:
            alignment_center = getattr(ft.alignment, 'CENTER', None)
        if alignment_center is None:
            container = ft.Container(content=card)
        else:
            container = ft.Container(content=card, alignment=alignment_center)
        self.main_view.controls.append(container)
        self.page.update()

    def build_dashboard(self):
        # safety guard
        if not self.current_user:
            self.build_login()
            return
        self.main_view.controls.clear()
        # ensure selector options are current (avoid updating before control in page)
        self.update_model_selector()

        header = ft.Row([
            ft.Column([ft.Text("MangoGuard", size=18, weight=ft.FontWeight.BOLD, color=self.primary), ft.Text(f"{self.current_user['username']}", size=12, color=self.muted)]),
            ft.Row([ft.IconButton(icon=Icons.DARK_MODE, on_click=self.toggle_theme, tooltip='Alternar tema'), ft.IconButton(icon=Icons.LOGOUT, tooltip="Cerrar sesión", on_click=self.logout)])
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        # Model status indicator (non-intrusive). If there are load errors, show a small 'Detalles' button.
        load_errors = get_model_load_errors()
        load_error_notice = None
        if load_errors:
            summaries = []
            details = []
            for k, v in load_errors.items():
                first_line = next((l for l in str(v).splitlines() if l.strip()), str(v))
                short = first_line
                if len(short) > 140:
                    short = short[:140] + '...'
                summaries.append(f"{k}: {short}")
                suggestion = ''
                if 'DepthwiseConv2D' in str(v) or 'groups' in str(v):
                    suggestion = 'Sugerencia: incompatibilidad con la versión de Keras/TensorFlow.'
                details.append(f"{k}: {v}\n{suggestion}")

            def _show_load_errors(e=None, txt='\n\n'.join(details)):
                dlg = ft.AlertDialog(title=ft.Text('Detalles de carga de modelos'), content=ft.Text(txt), actions=[ft.TextButton('Cerrar', on_click=lambda e: dlg.close())])
                self.page.dialog = dlg
                dlg.open = True
                self.page.update()

            # Subtle status: small text + non-intrusive button to view details
            load_error_notice = ft.Row([ft.Text("Algunos modelos no se cargaron correctamente", size=12, color=self.muted), ft.TextButton("Detalles", on_click=_show_load_errors)], alignment=ft.MainAxisAlignment.START)
        else:
            load_error_notice = ft.Row([ft.Text("Modelos cargados correctamente", size=12, color=self.muted)])


        upload_btn = ft.ElevatedButton("Seleccionar foto", icon=Icons.UPLOAD_FILE, on_click=self.open_file_picker, bgcolor=self.primary, color=Colors.WHITE)
        camera_btn = ft.ElevatedButton("Usar cámara", icon=Icons.PHOTO_CAMERA, on_click=self.open_file_picker)

        self.result_card = ft.Card(content=ft.Container(content=ft.Text("Sube una imagen para analizar", color=self.text_color), padding=18), elevation=1, color=self.card_bg)

        # history container reference (Column for internal list, so page scrolls naturally)
        self.history_container = ft.Column(spacing=8)

        model_row = ft.Row([
            ft.Text("Modelo:", color=self.text_color),
            ft.Container(self.model_selector, expand=True),
            ft.IconButton(icon=Icons.REFRESH, on_click=self.reload_models, tooltip='Recargar modelos')
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        actions_row = ft.Row([
            ft.ElevatedButton("Seleccionar foto", icon=Icons.UPLOAD_FILE, on_click=self.open_file_picker, bgcolor=self.primary, color=Colors.WHITE, expand=True),
            ft.ElevatedButton("Usar cámara", icon=Icons.PHOTO_CAMERA, on_click=self.open_file_picker, expand=True)
        ], spacing=12)

        controls = ft.Column([
            header,
            load_error_notice,
            self.picker_status,
            model_row,
            actions_row,
            self.result_card,
            ft.Divider(),
            ft.Text("Historial de análisis", size=16, weight=ft.FontWeight.BOLD, color=self.text_color),
            self.history_container
        ], spacing=14, expand=True)

        card = ft.Card(content=ft.Container(content=controls, padding=22, bgcolor=self.card_bg, border_radius=14), elevation=6)
        self.main_view.controls.append(card)
        self.update_history()
        self.page.update()

    def reload_models(self, e=None):
        try:
            load_models()
            self.update_model_selector()
            self.main_view.controls.append(ft.Text("Modelos recargados", color=self.primary))
        except Exception as ex:
            self.main_view.controls.append(ft.Text(f"Error recargando modelos: {ex}", color=Colors.RED))
        self.page.update()

    def open_file_picker(self, e=None):
        """Open client-side file picker (works on Web, Mobile, and Desktop)."""
        try:
            self.picker_status.value = "Abriendo selector..."
            self.page.update()
        except Exception:
            pass
        
        if self.file_picker:
            self.file_picker.pick_files(allow_multiple=False, file_type=ft.FilePickerFileType.IMAGE)
        else:
            # Fallback only if really needed (unlikely now)
            self.open_file_dialog_tk()
            
        try:
            self.picker_status.value = ""
            self.page.update()
        except Exception:
            pass
    def open_file_dialog_tk(self, e=None):
        """Fallback file selector using tkinter for desktop environments."""
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as ex:
            logger.warning("tkinter file dialog not available: %s", ex)
            self.main_view.controls.append(ft.Text("No es posible abrir un selector de archivos en esta plataforma.", color=Colors.RED))
            self.page.update()
            return

        def tk_worker():
            root = tk.Tk()
            root.withdraw()
            # Force window to top
            root.attributes('-topmost', True)
            root.focus_force()
            try:
                path = filedialog.askopenfilename(parent=root, filetypes=[("Imagenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")])
                if path:
                    self._upload_and_analyze(path, is_temp=False)
            finally:
                try:
                    root.destroy()
                except Exception:
                    pass

        Thread(target=tk_worker).start()

    def on_upload_progress(self, e: ft.FilePickerUploadEvent):
        if e.progress < 1.0:
            self.picker_status.value = f"Subiendo... {int(e.progress * 100)}%"
            self.page.update()
            return
        
        # Upload complete
        self.picker_status.value = "Procesando..."
        self.page.update()
        
        # We used the file_name from the event (the one we generated/sent? No, Flet might use the original or generated)
        # Actually Flet upload event returns the file_name relative to upload dir, usually matches safe_name if used in get_upload_url
        path = os.path.join(self.upload_dir, e.file_name)
        
        self._upload_and_analyze(path, is_temp=False)

    def file_picker_result(self, e: "ft.FilePickerResultEvent"):
        if not e.files:
            return
        picked = e.files[0]
        
        # Web/Mobile: We must upload the file to the server (local backend) to process it
        try:
           # Safety: Use UUID for the upload filename to avoid issues with spaces/emojis in mobile filenames
           ext = os.path.splitext(picked.name)[1]
           if not ext:
               ext = ".jpg"
           safe_name = f"{uuid4().hex}{ext}"

        # Web/Mobile: Use upload mechanism
        try:
           self.picker_status.value = "Generando URL de subida..."
           self.page.update()
           
           # Use simpler upload logic without fancy UUID if it complicates things, 
           # but UUID is safer. Keep UUID but log more status.
           safe_name = f"{uuid4().hex}_{picked.name}" 
           
           upload_url = self.page.get_upload_url(safe_name, 600)
           
           self.picker_status.value = "Iniciando transferencia..."
           self.page.update()
           
           self.file_picker.upload([
               ft.FilePickerUploadFile(
                   picked.name,
                   upload_url=upload_url
               )
           ])
           return
        except Exception as ex:
           self.picker_status.value = f"Error subida: {ex}"
           self.page.update()
           logger.error("Upload failed: %s", ex)
           # Fallback for desktop where upload might not be needed/configured or fails
           pass

        # Prefer local path when available (Desktop)
        path = getattr(picked, 'path', None)
        if path:
            self._upload_and_analyze(path, is_temp=False)
            return

        # Fallback: try to read bytes/content directly if upload logic skipped
        data = None
        for attr in ('bytes', 'content', 'data'):
            data = getattr(picked, attr, None)
            if data:
                break
        
        if not data:
            self.main_view.controls.append(ft.Text("Iniciando subida...", color=self.muted))
            self.page.update()
            return # Wait for on_upload if upload was triggered, else fail

        # Save to temp file
        ext = os.path.splitext(getattr(picked, 'name', 'upload'))[1] or '.jpg'
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        try:
            with os.fdopen(fd, 'wb') as out:
                out.write(data)
        except Exception as ex:
            self.main_view.controls.append(ft.Text(f"Error guardando archivo: {ex}", color=Colors.RED))
            self.page.update()
            return
        
        self._upload_and_analyze(tmp_path, is_temp=True)

    def update_history(self):
        if not self.current_user:
            return
        try:
            history = get_user_history(self.current_user['id'])
        except Exception:
            history = []
        content = []
        for h in history[:10]:
            lbl = h['result']['label']
            conf = h['result'].get('confidence', 0.0)
            model = h.get('model', 'heuristic')
            ts = h.get('timestamp', '')
            try:
                # Format timestamp nicer if possible
                dt = datetime.fromisoformat(ts)
                ts_str = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                ts_str = ts

            rec = h['result'].get('recommendation', {})
            rec_text = rec.get('recommendation', 'Sin recomendación')
            treat_text = rec.get('treatment', 'Sin tratamiento')
            meds = rec.get('medications', [])

            # Chips for meds
            chips = [ft.Chip(label=ft.Text(m, size=10), bgcolor=self.accent) for m in meds]

            # ExpansionTile for details
            item = ft.Card(
                content=ft.ExpansionTile(
                    leading=ft.Icon(Icons.HISTORY, color=self.primary),
                    title=ft.Text(f"{lbl} ({conf:.2f})", weight=ft.FontWeight.BOLD),
                    subtitle=ft.Text(f"{model} • {ts_str}", size=12, color=self.muted),
                    controls=[
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Recomendación:", weight=ft.FontWeight.BOLD, size=12),
                                ft.Text(rec_text, size=12),
                                ft.Divider(height=5, color="transparent"),
                                ft.Text("Tratamiento:", weight=ft.FontWeight.BOLD, size=12),
                                ft.Text(treat_text, size=12),
                                ft.Row(chips, wrap=True) if chips else ft.Container()
                            ], spacing=4),
                            padding=ft.padding.only(left=16, right=16, bottom=16)
                        )
                    ]
                ),
                elevation=1,
                color=self.card_bg
            )
            content.append(item)
            content.append(item)
        self.history_container.controls = content
        self.page.update()
    def _upload_and_analyze(self, path, is_temp=False):
        self.result_card.content = ft.Container(content=ft.Text("Analizando..."), padding=20)
        self.page.update()

        def worker():
            try:
                requested_model = self.model_selector.value or 'mango_v4'

                # If user explicitly selected a non-heuristic model that's not loaded, show helpful message and fallback
                available = available_models()
                from backend.models import model_file_status as _mfs
                files = _mfs()
                if requested_model != 'heuristic' and requested_model not in available:
                    # Subtle UX: polite notice and silent fallback to heuristic
                    if files.get(requested_model):
                        msg = "Modelo presente pero no cargado; se usará la heurística como fallback."
                    else:
                        msg = "Modelo no disponible; se usará la heurística como fallback."
                    self.result_card.content = ft.Container(content=ft.Column([ft.Text(msg, color=self.muted), ft.Text("Para ver más información, usa 'Ver detalles' en la sección de modelos." , size=12, color=self.muted)], spacing=6), padding=20)
                    try:
                        analysis = predict_image_sync(path, 'heuristic')
                        used_model = 'heuristic'
                    except Exception as he:
                        raise RuntimeError(f"Error al analizar la imagen: {he}")
                else:
                    # Try to analyze with requested model; if model isn't available or fails, fallback to heuristic
                    try:
                        analysis = predict_image_sync(path, requested_model)
                        used_model = requested_model
                    except Exception as model_ex:
                        logger.exception('Error en inferencia con modelo solicitado: %s', model_ex)
                        # If TensorFlow is not available or model file missing, fallback to heuristic predictor
                        try:
                            analysis = predict_image_sync(path, 'heuristic')
                            used_model = 'heuristic'
                        except Exception as he:
                            raise RuntimeError(f"Error al analizar la imagen: {model_ex} | fallback error: {he}")

                # Store locally if user present
                if self.current_user:
                    add_analysis(self.current_user['id'], self.current_user['username'], used_model, path, analysis)

                # Build UI
                meds = analysis.get('recommendation', {}).get('medications', [])
                chips = []
                for m in meds:
                    chips.append(ft.Container(ft.Text(m, size=12, color=self.text_color), padding=8, bgcolor=self.accent, border_radius=8, margin=ft.margin.only(right=6)))

                self.result_card.content = ft.Container(content=ft.Column([
                    ft.Row([ft.Text(f"Diagnóstico: {analysis['label']}", size=18, weight=ft.FontWeight.BOLD, color=self.primary), ft.Text(f"Modelo: {used_model}", size=12, color=self.muted)], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Text(f"Confianza: {analysis['confidence']:.2f}", color=self.muted),
                    ft.Text(f"Recomendación: {analysis['recommendation']['recommendation']}", color=self.text_color),
                    ft.Text(f"Tratamiento: {analysis['recommendation']['treatment']}", color=self.text_color),
                    ft.Row(chips),
                    ft.Image(src=path, width=420)
                ], spacing=10), padding=20)
                self.update_history()
            except Exception as ex:
                # Provide clearer UI guidance if model files / TF missing
                msg = str(ex)
                if 'TensorFlow' in msg or 'no cargado' in msg or 'not found' in msg:
                    hint = "(Nota: el modelo no está disponible; instala TensorFlow y asegúrate de que 'mango_model_v4_plus.h5' y 'class_indices.txt' estén en el directorio del proyecto.)"
                    self.result_card.content = ft.Container(content=ft.Column([
                        ft.Text("No se pudo usar el modelo seleccionado.", color=Colors.RED),
                        ft.Text(msg, color=self.muted),
                        ft.Text(hint, color=self.muted)
                    ], spacing=6), padding=20)
                else:
                    self.result_card.content = ft.Container(content=ft.Text(f"Error: {ex}"), padding=20)
            finally:
                if is_temp:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                self.page.update()

        Thread(target=worker).start()

    def login(self, e):
        try:
            email = (self.email.value or "").strip()
            password = (self.password.value or "")
            if not email or not password:
                raise Exception("Email y contraseña requeridos")
            user = authenticate_user(email, password)
            if not user:
                raise Exception("Credenciales inválidas")
            self.current_user = user
            save_current_user(user)
            self.build_dashboard()
        except Exception as ex:
            self.main_view.controls.append(ft.Text(f"Error al iniciar sesión: {ex}", color=Colors.RED))
            self.page.update()
    def register(self, e):
        try:
            username = (self.username.value or "").strip()
            email = (self.email.value or "").strip()
            password = (self.password.value or "")
            if not username or not email or not password:
                raise Exception("Todos los campos son obligatorios")
            user = create_user(username, email, password)
            # auto-login
            self.current_user = user
            save_current_user(user)
            self.build_dashboard()
        except Exception as ex:
            self.main_view.controls.append(ft.Text(f"Error al crear cuenta: {ex}", color=Colors.RED))
            self.page.update()

    def logout(self, e):
        self.current_user = None
        try:
            os.remove(AUTH_FILE)
        except OSError:
            pass
        self.build_login()


def main(page: ft.Page):
    # Configure basic logging for the UI process if not configured
    logging.basicConfig(level=logging.INFO)
    logger.info('Iniciando MangoApp')
    try:
        page.theme = ft.Theme(primary_color=PALETTE['primary'])
    except TypeError:
        try:
            t = ft.Theme()
            if hasattr(t, 'primary_color'):
                setattr(t, 'primary_color', PALETTE['primary'])
            elif hasattr(t, 'primary'):
                setattr(t, 'primary', PALETTE['primary'])
            page.theme = t
        except Exception:
            pass
    MangoApp(page)

if __name__ == '__main__':
    # Ensure uploads dir exists relative to CWD
    os.makedirs("uploads", exist_ok=True)
    ft.app(target=main, upload_dir="uploads")
