# mapeo sencillo para recomendaciones por etiqueta

RECOMMENDATIONS = {
    # Ejemplos — personalizar según etiquetas reales
    "nivel_0": {
        "stage": "Nivel 0",
        "diagnosis": "Fruto sano",
        "recommendation": "No se requiere tratamiento. Mantener buenas prácticas agrícolas.",
        "treatment": "N/A",
        "medications": []
    },
    "nivel_1": {
        "stage": "Nivel 1",
        "diagnosis": "Antracnosis en etapa inicial",
        "recommendation": "Eliminar frutos y hojas infectadas, mejorar ventilación y evitar encharcamientos.",
        "treatment": "Aplicar fungicidas protectores siguiendo indicaciones locales.",
        "medications": ["Captan", "Mancozeb", "Clorotalonil"]
    },
    "nivel_2": {
        "stage": "Nivel 2",
        "diagnosis": "Antracnosis moderada",
        "recommendation": "Podar partes afectadas, retirar residuos y mejorar circulación de aire.",
        "treatment": "Aplicar fungicida sistémico y repetir según ficha técnica. Considerar asesoría fitosanitaria.",
        "medications": ["Azoxystrobin", "Trifloxystrobin", "Propiconazole"]
    },
    "nivel_3": {
        "stage": "Nivel 3",
        "diagnosis": "Antracnosis avanzada",
        "recommendation": "Aislar la zona afectada y evitar la propagación.",
        "treatment": "Tratamiento intensivo con productos registrados y asesoría profesional.",
        "medications": ["Tebuconazole", "Azoxystrobin + Difenoconazole" ]
    },
    "nivel_4": {
        "stage": "Nivel 4",
        "diagnosis": "Antracnosis severa",
        "recommendation": "Retirar plantas severamente afectadas y aplicar un plan de manejo integrado.",
        "treatment": "Consulta con un fitosanitario y uso de productos autorizados según reglamentación local.",
        "medications": ["Tebuconazole", "Difenoconazole"]
    },
    "healthy": {
        "stage": "Sano",
        "diagnosis": "Fruto sano",
        "recommendation": "No se requiere tratamiento. Mantener buenas prácticas agrícolas.",
        "treatment": "N/A",
        "medications": []
    }
}


def get_recommendation(label: str) -> dict:
    """Return a recommendation dict for a given label (case-insensitive).

    The function tolerates numeric labels (e.g. '0' -> 'nivel_0'), variations like 'Nivel 1',
    or direct keys like 'nivel_2'. If no exact mapping exists returns a generic 'Desconocido' recommendation.
    """
    key = (label or '').strip().lower()
    if key.isdigit():
        key = f'nivel_{key}'
    # Normalize common variants
    key = key.replace('nivel ', 'nivel_').replace(' ', '_')
    return RECOMMENDATIONS.get(key, {
        "stage": "Desconocido",
        "diagnosis": label,
        "recommendation": "No se encontró una recomendación específica. Consulte con un fitosanitario.",
        "treatment": "Consultar recomendaciones locales.",
        "medications": []
    })
