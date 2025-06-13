"""
Versión web de la aplicación usando Streamlit
"""
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Configurar TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Configuración de la página
st.set_page_config(
    page_title="🐕 Clasificador de Emociones en Mascotas",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .happy { background-color: #90EE90; color: #006400; }
    .sad { background-color: #FFB6C1; color: #8B0000; }
    .angry { background-color: #FFA07A; color: #8B0000; }
    .scared { background-color: #DDA0DD; color: #4B0082; }
    .relaxed { background-color: #98FB98; color: #006400; }
    .anxious { background-color: #F0E68C; color: #8B8000; }
    .playful { background-color: #87CEEB; color: #000080; }
    .neutral { background-color: #D3D3D3; color: #2F4F4F; }
    
    .recommendation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Carga el modelo de emociones"""
    try:
        model = tf.keras.models.load_model("modelo_emociones.h5")
        return model
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocesa la imagen para el modelo"""
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar imagen
    image = image.resize(target_size)
    
    # Convertir a array numpy
    img_array = np.array(image)
    
    # Normalizar píxeles (0-1)
    img_array = img_array.astype('float32') / 255.0
    
    # Añadir dimensión batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_emotion_class(emotion):
    """Retorna la clase CSS para la emoción"""
    emotion_classes = {
        "Feliz": "happy",
        "Triste": "sad", 
        "Enojado": "angry",
        "Asustado": "scared",
        "Relajado": "relaxed",
        "Ansioso": "anxious",
        "Juguetón": "playful",
        "Neutral": "neutral"
    }
    return emotion_classes.get(emotion, "neutral")

def get_recommendations(emotion):
    """Devuelve recomendaciones basadas en la emoción"""
    recommendations_dict = {
        "Feliz": [
            "¡Tu mascota se ve muy contenta! Continúa con las actividades que la hacen feliz.",
            "Es un buen momento para jugar o dar un paseo.",
            "Considera tomar más fotos de estos momentos felices."
        ],
        "Triste": [
            "Tu mascota podría necesitar más atención y cariño.",
            "Verifica si hay cambios en su entorno que puedan afectarla.",
            "Considera consultar con un veterinario si persiste."
        ],
        "Enojado": [
            "Dale espacio a tu mascota y evita situaciones estresantes.",
            "Identifica qué pudo haber causado esta reacción.",
            "Usa técnicas de calma y refuerzo positivo."
        ],
        "Asustado": [
            "Proporciona un ambiente seguro y tranquilo.",
            "Evita ruidos fuertes o movimientos bruscos.",
            "Ofrece consuelo con voz suave y caricias gentiles."
        ],
        "Relajado": [
            "¡Perfecto! Tu mascota está en un estado ideal.",
            "Mantén el ambiente tranquilo y cómodo.",
            "Es un buen momento para el descanso."
        ],
        "Ansioso": [
            "Identifica y elimina posibles fuentes de estrés.",
            "Considera técnicas de relajación para mascotas.",
            "Mantén rutinas consistentes para reducir la ansiedad."
        ],
        "Juguetón": [
            "¡Es hora de jugar! Tu mascota tiene mucha energía.",
            "Prepara juguetes y actividades interactivas.",
            "Aprovecha para hacer ejercicio juntos."
        ],
        "Neutral": [
            "Tu mascota está en un estado normal y equilibrado.",
            "Observa si hay cambios en su comportamiento.",
            "Mantén las rutinas habituales de cuidado."
        ]
    }
    
    return recommendations_dict.get(emotion, ["Observa el comportamiento de tu mascota y consulta con un profesional si es necesario."])

def main():
    # Título principal
    st.markdown('<h1 class="main-header">🐕 Clasificador de Emociones en Mascotas</h1>', unsafe_allow_html=True)
    
    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información")
        st.write("Esta aplicación utiliza inteligencia artificial para detectar emociones en mascotas.")
        
        st.subheader("📋 Emociones detectables:")
        emotions = ["😊 Feliz", "😢 Triste", "😠 Enojado", "😨 Asustado", 
                   "😌 Relajado", "😰 Ansioso", "🎾 Juguetón", "😐 Neutral"]
        for emotion in emotions:
            st.write(f"• {emotion}")
        
        st.subheader("📸 Formatos soportados:")
        st.write("• PNG, JPG, JPEG")
        st.write("• Tamaño recomendado: 224x224px")
    
    # Cargar modelo
    model = load_model()
    
    if model is None:
        st.error("❌ No se pudo cargar el modelo. Verifica que 'modelo_emociones.h5' esté en el directorio.")
        return
    
    st.success("✅ Modelo cargado correctamente")
    
    # Área principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Cargar Imagen")
        uploaded_file = st.file_uploader(
            "Selecciona una imagen de tu mascota",
            type=['png', 'jpg', 'jpeg'],
            help="Sube una imagen clara de tu mascota para obtener mejores resultados"
        )
        
        if uploaded_file is not None:
            # Mostrar imagen
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_column_width=True)
            
            # Botón de análisis
            if st.button("🔍 Analizar Emoción", type="primary", use_container_width=True):
                with st.spinner("Analizando emoción..."):
                    try:
                        # Preprocesar imagen
                        processed_image = preprocess_image(image)
                        
                        # Hacer predicción
                        predictions = model.predict(processed_image, verbose=0)
                        probabilities = predictions[0]
                        
                        # Clases de emociones
                        emotion_classes = [
                            "Feliz", "Triste", "Enojado", "Asustado", 
                            "Relajado", "Ansioso", "Juguetón", "Neutral"
                        ]
                        
                        # Obtener resultado
                        predicted_class_idx = np.argmax(probabilities)
                        predicted_emotion = emotion_classes[predicted_class_idx]
                        confidence = probabilities[predicted_class_idx]
                        
                        # Guardar resultados en session state
                        st.session_state.emotion = predicted_emotion
                        st.session_state.confidence = confidence
                        st.session_state.probabilities = probabilities
                        st.session_state.emotion_classes = emotion_classes
                        
                    except Exception as e:
                        st.error(f"Error en el análisis: {e}")
    
    with col2:
        st.subheader("📊 Resultados")
        
        if hasattr(st.session_state, 'emotion'):
            # Mostrar emoción principal
            emotion_class = get_emotion_class(st.session_state.emotion)
            st.markdown(
                f'<div class="emotion-result {emotion_class}">🎯 {st.session_state.emotion}</div>',
                unsafe_allow_html=True
            )
            
            # Mostrar confianza
            st.metric("Confianza", f"{st.session_state.confidence:.1%}")
            
            # Gráfico de barras con todas las probabilidades
            st.subheader("📈 Todas las Probabilidades")
            prob_data = {}
            for emotion, prob in zip(st.session_state.emotion_classes, st.session_state.probabilities):
                prob_data[emotion] = float(prob)
            
            st.bar_chart(prob_data)
            
            # Recomendaciones
            st.subheader("💡 Recomendaciones")
            recommendations = get_recommendations(st.session_state.emotion)
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(
                    f'<div class="recommendation-box"><strong>{i}.</strong> {rec}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("👆 Carga una imagen y haz clic en 'Analizar Emoción' para ver los resultados")

if __name__ == "__main__":
    main()
