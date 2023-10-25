import os
import pandas as pd

OUTPUT_DIR = './train_landmark_files'
MAPPING_FILE_PATH = 'parquet_to_sign_mapping.csv'

# Cargar el mapeo de parquet a signo desde el archivo CSV
mapping_df = pd.read_csv(MAPPING_FILE_PATH)

# Lista para almacenar los datos
data_list = []

# Función para obtener el participant_id y sequence_id del path
def extract_ids_from_path(file_path):
    _, participant_id, sequence_id = file_path.split(os.sep)[-3:]
    sequence_id = sequence_id.split('.')[0]  # Eliminar la extensión .parquet
    return participant_id, sequence_id

# Recorrer el directorio de salida
for root, dirs, files in os.walk(OUTPUT_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        participant_id, sequence_id = extract_ids_from_path(file_path)
        # Obtener el signo del archivo CSV
        sign = mapping_df.loc[mapping_df['parquet_file'] == sequence_id + '.parquet', 'sign'].values[0]
        # Agregar los datos a la lista
        data_list.append({
            'path': file_path,
            'participant_id': participant_id,
            'sequence_id': sequence_id,
            'sign': sign
        })

# Crear un DataFrame a partir de la lista de datos
df = pd.DataFrame(data_list)

# Guardar el DataFrame como un archivo CSV
df.to_csv('output.csv', index=False)
