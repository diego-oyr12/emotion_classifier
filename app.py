import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Emociones en Mascotas",
    page_icon="[DOG]",
    layout="wide"
)

# Clases de emociones (ajusta según tu modelo)
EMOTION_CLASSES = [
    "Feliz", "Triste", "Enojado", "Asustado", 
    "Relajado", "Ansioso", "Juguetón", "Neutral"
]

@st.cache_resource
def load_model():
    """Carga el modelo de emociones"""
    try:
        model_path = "modelo_emociones.h5"
        if not os.path.exists(model_path):
            st.error(f"No se encontró el archivo del modelo: {model_path}")
            st.info("Asegúrate de que el archivo 'modelo_emociones.h5' esté en el directorio de la aplicación")
            return None
        
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocesa la imagen para el modelo"""
    try:
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
    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")
        return None

def predict_emotion(model, processed_image):
    """Realiza la predicción de emoción"""
    try:
        # Hacer predicción
        predictions = model.predict(processed_image, verbose=0)
        
        # Obtener probabilidades
        probabilities = predictions[0]
        
        # Obtener clase predicha
        predicted_class_idx = np.argmax(probabilities)
        predicted_emotion = EMOTION_CLASSES[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        return predicted_emotion, confidence, probabilities
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        return None, None, None

def main():
    st.title("[DOG] Clasificador de Emociones en Mascotas")
    st.markdown("---")
    
    # Cargar modelo
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.write(f"**Arquitectura:** {model.name if hasattr(model, 'name') else 'Modelo personalizado'}")
        st.write(f"**Clases de emociones:** {', '.join(EMOTION_CLASSES)}")
        st.write(f"**Forma de entrada:** {model.input_shape}")
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración")
    
    # Ajustar umbral de confianza
    confidence_threshold = st.sidebar.slider(
        "Umbral de confianza mínima",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Mostrar todas las probabilidades
    show_all_probs = st.sidebar.checkbox("Mostrar todas las probabilidades", value=True)
    
    # Área principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("[CAMERA] Cargar Imagen")
        
        # Opciones de carga
        upload_option = st.radio(
            "Selecciona una opción:",
            ["Subir archivo", "Usar cámara web"]
        )
        
        uploaded_file = None
        
        if upload_option == "Subir archivo":
            uploaded_file = st.file_uploader(
                "Selecciona una imagen de tu mascota",
                type=['png', 'jpg', 'jpeg'],
                help="Formatos soportados: PNG, JPG, JPEG"
            )
        else:
            uploaded_file = st.camera_input("Toma una foto de tu mascota")
        
        if uploaded_file is not None:
            # Mostrar imagen
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_column_width=True)
            
            # Información de la imagen
            st.info(f"**Dimensiones:** {image.size[0]} x {image.size[1]} píxeles")
    
    with col2:
        st.header("[TARGET] Resultados de Clasificación")
        
        if uploaded_file is not None:
            with st.spinner("Analizando emoción..."):
                # Procesar imagen
                processed_image = preprocess_image(image)
                
                if processed_image is not None:
                    # Hacer predicción
                    emotion, confidence, probabilities = predict_emotion(model, processed_image)
                    
                    if emotion is not None:
                        # Mostrar resultado principal
                        if confidence >= confidence_threshold:
                            st.success(f"**Emoción detectada:** {emotion}")
                            st.metric("Confianza", f"{confidence:.2%}")
                        else:
                            st.warning(f"**Emoción detectada:** {emotion} (Baja confianza)")
                            st.metric("Confianza", f"{confidence:.2%}")
                        
                        # Mostrar todas las probabilidades
                        if show_all_probs:
                            st.subheader("[CHART] Todas las Probabilidades")
                            
                            # Crear DataFrame para mejor visualización
                            prob_data = []
                            for i, (emotion_class, prob) in enumerate(zip(EMOTION_CLASSES, probabilities)):
                                prob_data.append({
                                    "Emoción": emotion_class,
                                    "Probabilidad": f"{prob:.2%}",
                                    "Valor": prob
                                })
                            
                            # Ordenar por probabilidad
                            prob_data.sort(key=lambda x: x["Valor"], reverse=True)
                            
                            # Mostrar barras de progreso
                            for item in prob_data:
                                st.write(f"**{item['Emoción']}:** {item['Probabilidad']}")
                                st.progress(item["Valor"])
                        
                        # Recomendaciones basadas en la emoción
                        st.subheader("[BULB] Recomendaciones")
                        recommendations = get_recommendations(emotion)
                        for rec in recommendations:
                            st.write(f"• {rec}")
        else:
            st.info("👆 Carga una imagen para comenzar el análisis")

def get_recommendations(emotion):
    """Devuelve recomendaciones basadas en la emoción detectada"""
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

if __name__ == "__main__":
    main()
