"""
Script para probar que el modelo funciona correctamente
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import os

def test_model_loading():
    """Prueba la carga del modelo"""
    print("[SEARCH] Probando carga del modelo...")
    
    try:
        if not os.path.exists("modelo_emociones.h5"):
            print("[X] Error: No se encontró modelo_emociones.h5")
            return False
        
        model = tf.keras.models.load_model("modelo_emociones.h5")
        print("[CHECK] Modelo cargado exitosamente")
        print(f"[CHART] Forma de entrada: {model.input_shape}")
        print(f"[CHART] Forma de salida: {model.output_shape}")
        return model
    except Exception as e:
        print(f"[X] Error al cargar modelo: {e}")
        return False

def test_prediction(model):
    """Prueba una predicción con imagen sintética"""
    print("\n🧪 Probando predicción...")
    
    try:
        # Crear imagen de prueba
        input_shape = model.input_shape[1:3]  # (height, width)
        test_image = np.random.randint(0, 255, (*input_shape, 3), dtype=np.uint8)
        test_image = test_image.astype('float32') / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        
        # Hacer predicción
        prediction = model.predict(test_image, verbose=0)
        print(f"[CHECK] Predicción exitosa")
        print(f"[CHART] Forma de salida: {prediction.shape}")
        print(f"[CHART] Suma de probabilidades: {np.sum(prediction[0]):.4f}")
        
        return True
    except Exception as e:
        print(f"[X] Error en predicción: {e}")
        return False

def main():
    print("[ROCKET] Iniciando pruebas del modelo...")
    print("=" * 50)
    
    # Probar carga del modelo
    model = test_model_loading()
    if not model:
        return
    
    # Probar predicción
    if test_prediction(model):
        print("\n[CHECK] Todas las pruebas pasaron exitosamente!")
        print("[PARTY] Tu modelo está listo para usar en la aplicación")
    else:
        print("\n[X] Algunas pruebas fallaron")
        print("🔧 Revisa tu modelo antes de usar la aplicación")

if __name__ == "__main__":
    main()
