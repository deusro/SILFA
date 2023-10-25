import os
import pyarrow as pa
import pyarrow.parquet as pq
import mediapipe as mp
import cv2
import pandas as pd
mp_holistic = mp.solutions.holistic

DATA_DIR = './data'
OUTPUT_DIR = './train_landmark_files/100'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Crear el archivo CSV para almacenar la relación
csv_filename = 'parquet_to_sign_mapping.csv'
csv_path = os.path.join(OUTPUT_DIR, csv_filename)
csv_data = []

for dir_ in os.listdir(DATA_DIR):
    for img_idx, img_path in enumerate(os.listdir(os.path.join(DATA_DIR, dir_))):
        data_aux = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with mp_holistic.Holistic(static_image_mode=True) as holistic:
            results = holistic.process(img_rgb)

            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    data_aux.append({'frame': 1, 'row_id': f'{1}-pose-{idx}', 'type': 'pose', 'x': landmark.x, 'y': landmark.y, 'z': landmark.z if landmark.HasField('z') else 0.0})

            if results.face_landmarks:
                for idx, landmark in enumerate(results.face_landmarks.landmark):
                    data_aux.append({'frame': 1, 'row_id': f'{1}-face-{idx}', 'type': 'face', 'x': landmark.x, 'y': landmark.y, 'z': landmark.z if landmark.HasField('z') else 0.0})

            if results.left_hand_landmarks:
                for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                    data_aux.append({'frame': 1, 'row_id': f'{1}-left_hand-{idx}', 'type': 'left_hand', 'x': landmark.x, 'y': landmark.y, 'z': landmark.z if landmark.HasField('z') else 0.0})

            if results.right_hand_landmarks:
                for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                    data_aux.append({'frame': 1, 'row_id': f'{1}-right_hand-{idx}', 'type': 'right_hand', 'x': landmark.x, 'y': landmark.y, 'z': landmark.z if landmark.HasField('z') else 0.0})

            # Dibujar landmarks en la imagen
            
        # Convertir datos a Pandas y luego a una tabla Arrow
        panda_aux = pd.DataFrame(data_aux)
        table = pa.Table.from_pandas(panda_aux)

        # Escribir la tabla Arrow en Parquet
        output_path = os.path.join(OUTPUT_DIR, f'{img_idx + 1000}.parquet')
        pq.write_table(table, output_path)

        # Añadir la entrada al archivo CSV
        csv_data.append({'parquet_file': f'{img_idx + 1000}.parquet', 'sign': dir_})

# Guardar el archivo CSV
df_csv = pd.DataFrame(csv_data)
df_csv.to_csv(csv_path, index=False)

