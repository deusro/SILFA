import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Inicializar la instancia de MediaPipe Holistic
holistic = mp_holistic.Holistic()

# Cargar el modelo de TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path='SILFA/model.tflite')
interpreter.allocate_tensors()

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB (MediaPipe usa imágenes RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Obtener los resultados de MediaPipe
    results = holistic.process(rgb_frame)

    # Verificar si se detectaron todas las partes necesarias
    if results.pose_landmarks and results.face_landmarks and results.left_hand_landmarks and results.right_hand_landmarks:
        # Inicializar lista para almacenar landmarks
        all_landmarks = []

        # Obtener landmarks de la pose
        pose_landmarks = results.pose_landmarks.landmark
        all_landmarks.extend([(landmark.x, landmark.y, landmark.z if landmark.HasField('z') else 0.0) for landmark in pose_landmarks])

        # Obtener landmarks de la cara
        face_landmarks = results.face_landmarks.landmark
        all_landmarks.extend([(landmark.x, landmark.y, landmark.z if landmark.HasField('z') else 0.0) for landmark in face_landmarks])

        # Obtener landmarks de la mano izquierda
        left_hand_landmarks = results.left_hand_landmarks.landmark
        all_landmarks.extend([(landmark.x, landmark.y, landmark.z if landmark.HasField('z') else 0.0) for landmark in left_hand_landmarks])

        # Obtener landmarks de la mano derecha
        right_hand_landmarks = results.right_hand_landmarks.landmark
        all_landmarks.extend([(landmark.x, landmark.y, landmark.z if landmark.HasField('z') else 0.0) for landmark in right_hand_landmarks])

        # Convertir a un numpy array
        landmarks_array = np.array(all_landmarks).flatten()

        # Ajustar las dimensiones según el formato esperado por el modelo
        landmarks_array = landmarks_array.reshape((1, 543, 3))

        # Eliminar NaN si los hay
        landmarks_array = landmarks_array[~np.isnan(landmarks_array).any(axis=(1, 2))]

        if landmarks_array.shape[0] == 1:  # Asegurarse de que haya al menos un landmark detectado
            # Aquí puedes hacer más procesamiento según tus necesidades

            # Preprocesar y alimentar los landmarks al modelo
            input_tensor_index = interpreter.get_input_details()[0]['index']
            interpreter.set_tensor(input_tensor_index, landmarks_array.astype(np.float32))
            interpreter.invoke()

            # Obtener la salida del modelo
            output_tensor_index = interpreter.get_output_details()[0]['index']
            model_output = interpreter.get_tensor(output_tensor_index)

            # Aquí puedes usar model_output para obtener la predicción final
            # Por ejemplo, imprimir la clase predicha
            predicted_class = np.argmax(model_output)
            print(f'Predicted Class: {predicted_class}')

            # Dibujar landmarks en el frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.face_landmarks)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks)

    # Mostrar el frame
    cv2.imshow('Holistic Model', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
