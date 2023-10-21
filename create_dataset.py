import os
import pyarrow as pa
import pyarrow.parquet as pq
import mediapipe as mp
import cv2
import pandas as pd
mp_holistic = mp.solutions.holistic

DATA_DIR = './data'
OUTPUT_DIR = './train_landmark_files'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
            

        # Convert data to Pandas and the to a  Arrow Table
        panda_aux = pd.DataFrame(data_aux)
        table = pa.Table.from_pandas(panda_aux)
        # See pandas
        print(panda_aux.head)
        # Cheking types
        unique_types = panda_aux["type"].nunique()
        print(unique_types)
        types_in_video = panda_aux["type"].unique()
        print(types_in_video)

        # Write Arrow table to Parquet
        output_path = os.path.join(OUTPUT_DIR, f'NombreEstatico{img_idx + 1}.parquet')
        pq.write_table(table, output_path)

        ##AND NOW I HAVE BECOME DEAD, DESTROYER OF WORDS##

