import os
import flet as ft
from frontend.app import main

if __name__ == '__main__':
    # Ensure uploads dir exists relative to CWD
    os.makedirs("uploads", exist_ok=True)
    
    # Run the app
    # export_asgi_app is not needed for APK, just standard ft.app
    ft.app(target=main, upload_dir="uploads", secret_key="mangoguard_secret_key_123")
