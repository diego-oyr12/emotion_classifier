# Clasificador de Emociones en Mascotas 🐕

Una aplicación de escritorio para clasificar emociones en mascotas usando inteligencia artificial.

## Características

- ✅ Usa tu modelo entrenado real (`modelo_emociones.h5`)
- 🖼️ Interfaz gráfica intuitiva con Streamlit
- 📸 Soporte para carga de archivos y cámara web
- 📊 Visualización de probabilidades detalladas
- 💡 Recomendaciones basadas en emociones detectadas
- 📦 Fácil empaquetado y distribución

## Instalación

### Opción 1: Instalación directa
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Opción 2: Instalación como paquete
\`\`\`bash
pip install -e .
\`\`\`

## Uso

1. **Coloca tu modelo:** Asegúrate de que `modelo_emociones.h5` esté en el directorio raíz
2. **Ejecuta la aplicación:**
   \`\`\`bash
   streamlit run app.py
   \`\`\`
3. **Abre tu navegador:** Ve a `http://localhost:8501`

## Empaquetado

### Para Windows (usando PyInstaller)
\`\`\`bash
pip install pyinstaller
pyinstaller --onefile --add-data "modelo_emociones.h5;." app.py
\`\`\`

### Para crear un ejecutable portable
\`\`\`bash
pip install auto-py-to-exe
auto-py-to-exe
\`\`\`

### Para distribución con Docker
\`\`\`bash
docker build -t pet-emotion-classifier .
docker run -p 8501:8501 pet-emotion-classifier
\`\`\`

## Estructura del Proyecto

\`\`\`
pet-emotion-classifier/
├── app.py                 # Aplicación principal
├── modelo_emociones.h5    # Tu modelo entrenado
├── requirements.txt       # Dependencias
├── setup.py              # Configuración del paquete
├── README.md             # Este archivo
└── Dockerfile            # Para containerización
\`\`\`

## Personalización

### Ajustar clases de emociones
Modifica la lista `EMOTION_CLASSES` en `app.py`:
```python
EMOTION_CLASSES = [
    "Tu_Emocion_1", "Tu_Emocion_2", ...
]
