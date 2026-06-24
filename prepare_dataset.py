# Steps:
# 1) Load filelist of audio that needs to be extracted
# 2) For each audio file, extract features with padding, then remove padding from features
# 3) Save features to CSV file

import numpy as np
import pandas as pd
import os
import feature_extractor as fe
import feature_selection as fs
from sklearn.decomposition import PCA
from scipy.stats import entropy
from umap import UMAP #pip install umap-learn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans

feature_extractor = fe.FeatureExtractor('hubert_l')

def load_filelist(file_path):
    """
    Loads a list of filenames from a text file.
    
    Args:
        file_path: Path to the text file containing the list of filenames.
    Returns:
        A list of filenames.
    """
    if not os.path.exists(file_path):
        print(f"Error: File list not found at {file_path}")
        return []
    
    with open(file_path, 'r') as f:
        filelist_items = [line.strip() for line in f if line.strip()]
    
    return filelist_items

def extract_and_save_features(filelist, data_directory, feature_output_dir):
    """
    Extracts features from audio files and saves them as .npy files.
    
    Args:
        filelist: A list of filenames to process.
        data_directory: Directory where the audio files are located.
        feature_output_dir: Directory where the extracted features will be saved.
    """

    for filename in filelist:
        # Only process if features do not already exist
        if not os.path.exists(os.path.join(feature_output_dir, filename.replace('.wav', '_features.npy'))):
            file_path = os.path.join(data_directory, filename)
            print(f"Processing file: {file_path}")
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping.")
                continue

            # Extract features
            features = feature_extractor.get_24th_layer_features_averages(
                file_path, extract_with_padding=True
            ).flatten()
            np.save(os.path.join(feature_output_dir, filename.replace('.wav', '_features.npy')), features)

def process_challenge_data(input_folder, output_folder):
     """
     Procesa todos los audios de una carpeta, extrae características HuBERT,
     las reduce a 103 dimensiones y las guarda con el formato EN_xxx_x.npy
     """
     if not os.path.exists(output_folder):
        os.makedirs(output_folder)

     # Listar archivos .wav que empiecen con ES_
     audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and f.startswith('ES_')]
    
     print(f"Encontrados {len(audio_files)} archivos para procesar.")

     for filename in audio_files:
         path = os.path.join(input_folder, filename)
        
         # 1. Extraer características de la capa 24 (Dimensión 1024)
         # Usamos tu método existente
         feat_1024 = feature_extractor.get_24th_layer_features_averages(path, extract_with_padding=True)
        
         # 2. Reducir a 103 dimensiones usando 'extract_winners'
         # Según las instrucciones, para el target de inglés se usan 103 dimensiones
         feat_103 = fs.extract_winners(feat_1024, "english")
        
         # 3. Asegurar que la forma sea (103,)
         feat_103 = np.array(feat_103, dtype=np.float64).flatten()
         #print( feat_103)
         
         if feat_103.shape != (103,):
            print(f"Error en dimensión para {filename}: {feat_103.shape}")
            continue

         # 4. Cambiar nombre de ES_xxx_x.wav a EN_xxx_x.npy
         new_filename = filename.replace("ES_", "EN_").replace(".wav", "_features.npy")
         output_path = os.path.join(output_folder, new_filename)

         
         # 5. Guardar como archivo numpy
         np.save(output_path, feat_103)
         print(f"Procesado: {filename} -> {new_filename}") 

def process_with_pca(input_folder, output_folder):
    """
    1. Extrae características de 1024 dimensiones de todos los audios.
    2. Aplica PCA para reducir todas a 103 dimensiones.
    3. Guarda los archivos con el nombre correcto.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and f.startswith('ES_')]
    
    all_features_1024 = []
    filenames_order = []

    print(f"Paso 1: Extrayendo características originales de {len(audio_files)} archivos...")

    for filename in audio_files:
        path = os.path.join(input_folder, filename)
        # Extraer vector de 1024
        feat_1024 = feature_extractor.get_24th_layer_features_averages(path, extract_with_padding=True)
        
        all_features_1024.append(feat_1024)
        filenames_order.append(filename)

    # Convertir a matriz de numpy (Número de audios, 1024)
    data_matrix = np.array(all_features_1024)

    print(f"Paso 2: Aplicando PCA para reducir de 1024 a 103 dimensiones...")
    
    # Inicializar y ajustar PCA
    # Nota: El número de muestras debe ser mayor a 103 para que esto funcione bien
    pca = PCA(n_components=103)
    data_reduced = pca.fit_transform(data_matrix)

    print(f"Paso 3: Guardando archivos .npy...")

    for i, filename in enumerate(filenames_order):
        # Obtener el vector de 103 dimensiones
        feat_103 = data_reduced[i] # Esto ya tiene forma (103,)

        # Cambiar nombre: ES_xxx_x.wav -> EN_xxx_x.npy
        new_filename = filename.replace("ES_", "EN_").replace(".wav", "_features.npy")
        output_path = os.path.join(output_folder, new_filename)
        feat_103 = np.array(feat_103, dtype=np.float64).flatten()
        np.save(output_path, feat_103)

    print(f"¡Hecho! Todos los archivos guardados en: {output_folder}")

def process_with_UMAP(input_folder, output_folder):
    """
    1. Extrae características de 1024 dimensiones de todos los audios.
    2. Aplica PCA para reducir todas a 103 dimensiones.
    3. Guarda los archivos con el nombre correcto.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and f.startswith('ES_')]
    
    all_features_1024 = []
    filenames_order = []

    print(f"Paso 1: Extrayendo características originales de {len(audio_files)} archivos...")

    for filename in audio_files:
        path = os.path.join(input_folder, filename)
        # Extraer vector de 1024
        feat_1024 = feature_extractor.get_24th_layer_features_averages(path, extract_with_padding=True)
        
        all_features_1024.append(feat_1024)
        filenames_order.append(filename)

    # Convertir a matriz de numpy (Número de audios, 1024)
    data_matrix = np.array(all_features_1024)

    umap_reducer = UMAP(n_components=103, random_state=42)
    
    data_reduced  = umap_reducer.fit_transform(data_matrix)
   

    print(f"Paso 3: Guardando archivos .npy...")

    for i, filename in enumerate(filenames_order):
        # Obtener el vector de 103 dimensiones
        feat_103 = data_reduced[i] # Esto ya tiene forma (103,)

        # Cambiar nombre: ES_xxx_x.wav -> EN_xxx_x.npy
        new_filename = filename.replace("ES_", "EN_").replace(".wav", "_features.npy")
        output_path = os.path.join(output_folder, new_filename)
        feat_103 = np.array(feat_103, dtype=np.float64).flatten()
        np.save(output_path, feat_103)

    print(f"¡Hecho! Todos los archivos guardados en: {output_folder}")

def process_with_KBest(input_folder, output_folder):
    """
    1. Extrae características de 1024 dimensiones de todos los audios.
    2. Aplica PCA para reducir todas a 103 dimensiones.
    3. Guarda los archivos con el nombre correcto.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and f.startswith('ES_')]
    
    all_features_1024 = []
    filenames_order = []

    print(f"Paso 1: Extrayendo características originales de {len(audio_files)} archivos...")

    for filename in audio_files:
        path = os.path.join(input_folder, filename)
        # Extraer vector de 1024
        feat_1024 = feature_extractor.get_24th_layer_features_averages(path, extract_with_padding=True)
        
        all_features_1024.append(feat_1024)
        filenames_order.append(filename)

    # Convertir a matriz de numpy (Número de audios, 1024)
    data_matrix = np.array(all_features_1024)

    # Supongamos que tienes etiquetas (si no, usa Varianza)
    selector = SelectKBest(k=103, score_func=lambda X, y: np.var(X, axis=0))
    
    data_reduced  = selector.fit_transform(data_matrix)  # si no tienes etiquetas
    
    print(f"Paso 3: Guardando archivos .npy...")

    for i, filename in enumerate(filenames_order):
        # Obtener el vector de 103 dimensiones
        feat_103 = data_reduced[i] # Esto ya tiene forma (103,)

        # Cambiar nombre: ES_xxx_x.wav -> EN_xxx_x.npy
        new_filename = filename.replace("ES_", "EN_").replace(".wav", "_features.npy")
        output_path = os.path.join(output_folder, new_filename)
        feat_103 = np.array(feat_103, dtype=np.float64).flatten()
        np.save(output_path, feat_103)

    print(f"¡Hecho! Todos los archivos guardados en: {output_folder}")

def process_with_Centroid(input_folder, output_folder):
    """
    1. Extrae características de 1024 dimensiones de todos los audios.
    2. Aplica PCA para reducir todas a 103 dimensiones.
    3. Guarda los archivos con el nombre correcto.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and f.startswith('ES_')]
    
    all_features_1024 = []
    filenames_order = []

    print(f"Paso 1: Extrayendo características originales de {len(audio_files)} archivos...")

    for filename in audio_files:
        path = os.path.join(input_folder, filename)
        # Extraer vector de 1024
        feat_1024 = feature_extractor.get_24th_layer_features_averages(path, extract_with_padding=True)
        
        all_features_1024.append(feat_1024)
        filenames_order.append(filename)

    # Convertir a matriz de numpy (Número de audios, 1024)
    data_matrix = np.array(all_features_1024)

    kmeans = KMeans(n_clusters=103, random_state=42)
    
    data_reduced= kmeans.fit_transform(data_matrix)  # [1, 103]
   
  
    print(f"Paso 3: Guardando archivos .npy...")

    for i, filename in enumerate(filenames_order):
        # Obtener el vector de 103 dimensiones
        feat_103 = data_reduced[i] # Esto ya tiene forma (103,)

        # Cambiar nombre: ES_xxx_x.wav -> EN_xxx_x.npy
        new_filename = filename.replace("ES_", "EN_").replace(".wav", "_features.npy")
        output_path = os.path.join(output_folder, new_filename)
        feat_103 = np.array(feat_103, dtype=np.float64).flatten()
        np.save(output_path, feat_103)

    print(f"¡Hecho! Todos los archivos guardados en: {output_folder}")


def process_challenge_data1(input_folder, output_folder):
     """
     Procesa todos los audios de una carpeta, extrae características HuBERT,
     las reduce a 103 dimensiones y las guarda con el formato EN_xxx_x.npy
     """
     if not os.path.exists(output_folder):
        os.makedirs(output_folder)

     # Listar archivos .wav que empiecen con ES_
     audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and f.startswith('EN_')]
    
     print(f"Encontrados {len(audio_files)} archivos para procesar.")

     for filename in audio_files:
         path = os.path.join(input_folder, filename)
        
         # 1. Extraer características de la capa 24 (Dimensión 1024)
         # Usamos tu método existente
         feat_1024 = feature_extractor.get_24th_layer_features_averages(path, extract_with_padding=True)
        
         # 2. Reducir a 103 dimensiones usando 'extract_winners'
         # Según las instrucciones, para el target de inglés se usan 103 dimensiones
         feat_101 = fs.extract_winners(feat_1024, "spanish")
        
         # 3. Asegurar que la forma sea (103,)
         feat_101 = np.array(feat_101, dtype=np.float64).flatten()
         #print( feat_103)
         
         if feat_101.shape != (101,):
            print(f"Error en dimensión para {filename}: {feat_101.shape}")
            continue

         # 4. Cambiar nombre de ES_xxx_x.wav a EN_xxx_x.npy
         new_filename = filename.replace("EN_", "ES_").replace(".wav", "_features.npy")
         output_path = os.path.join(output_folder, new_filename)

         
         # 5. Guardar como archivo numpy
         np.save(output_path, feat_101)
         print(f"Procesado: {filename} -> {new_filename}") 

def process_with_pca1(input_folder, output_folder):
    """
    1. Extrae características de 1024 dimensiones de todos los audios.
    2. Aplica PCA para reducir todas a 103 dimensiones.
    3. Guarda los archivos con el nombre correcto.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and f.startswith('EN_')]
    
    all_features_1024 = []
    filenames_order = []

    print(f"Paso 1: Extrayendo características originales de {len(audio_files)} archivos...")

    for filename in audio_files:
        path = os.path.join(input_folder, filename)
        # Extraer vector de 1024
        feat_1024 = feature_extractor.get_24th_layer_features_averages(path, extract_with_padding=True)
        
        all_features_1024.append(feat_1024)
        filenames_order.append(filename)

    # Convertir a matriz de numpy (Número de audios, 1024)
    data_matrix = np.array(all_features_1024)

    print(f"Paso 2: Aplicando PCA para reducir de 1024 a 103 dimensiones...")
    
    # Inicializar y ajustar PCA
    # Nota: El número de muestras debe ser mayor a 103 para que esto funcione bien
    pca = PCA(n_components=101)
    data_reduced = pca.fit_transform(data_matrix)

    print(f"Paso 3: Guardando archivos .npy...")

    for i, filename in enumerate(filenames_order):
        # Obtener el vector de 103 dimensiones
        feat_101 = data_reduced[i] # Esto ya tiene forma (103,)

        # Cambiar nombre: ES_xxx_x.wav -> EN_xxx_x.npy
        new_filename = filename.replace("EN_", "ES_").replace(".wav", "_features.npy")
        output_path = os.path.join(output_folder, new_filename)
        feat_101 = np.array(feat_101, dtype=np.float64).flatten()
        np.save(output_path, feat_101)

    print(f"¡Hecho! Todos los archivos guardados en: {output_folder}")

def calculate_entropy(matrix):
    """
    Calcula la entropía de Shannon para cada columna (rasgo) de la matriz.
    """
    # Agregamos una pequeña constante para evitar log(0)
    # Primero normalizamos cada columna para que parezca una distribución de probabilidad
    matrix_abs = np.abs(matrix)
    pk = matrix_abs / matrix_abs.sum(axis=0, keepdims=True)
    return entropy(pk)

def process_with_entropy_selection(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and f.startswith('ES_')]
    
    all_features_1024 = []
    filenames_order = []

    # PASO 1: Extraer los vectores originales (1024) de todos los archivos
    print(f"Extrayendo características de {len(audio_files)} archivos...")
    for filename in audio_files:
        path = os.path.join(input_folder, filename)
        feat_1024 = feature_extractor.get_24th_layer_features_averages(path, extract_with_padding=True)
        all_features_1024.append(feat_1024)
        filenames_order.append(filename)
        print(filename)

    data_matrix = np.array(all_features_1024) # Forma: (N_audios, 1024)

    # PASO 2: Calcular Entropía por dimensión
    print("Calculando entropía de las 1024 dimensiones...")
    entropies = calculate_entropy(data_matrix)

    # PASO 3: Seleccionar los índices de las 103 dimensiones con mayor entropía
    # argsort da los índices de menor a mayor, por eso usamos [-103:]
    best_indices = np.argsort(entropies)[-103:]
    
    print(f"Dimensiones seleccionadas basadas en alta entropía: {best_indices[:5]}... (total 103)")

    # PASO 4: Filtrar y Guardar
    for i, filename in enumerate(filenames_order):
        # Seleccionamos solo las 103 dimensiones ganadoras
        feat_103 = data_matrix[i, best_indices]
        
        # Forzamos formato float64 y forma (103,) como discutimos antes
        feat_103 = feat_103.astype(np.float64).flatten()

        new_filename = filename.replace("ES_", "EN_").replace(".wav", "_features.npy")
        output_path = os.path.join(output_folder, new_filename)
        
        np.save(output_path, feat_103)
        print(f"Procesado: {filename} -> {new_filename}") 
        
    print(f"¡Proceso completado! Archivos guardados en {output_folder}")    
    
def process_with_entropy_selection1(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and f.startswith('EN_')]
    
    all_features_1024 = []
    filenames_order = []

    # PASO 1: Extraer los vectores originales (1024) de todos los archivos
    print(f"Extrayendo características de {len(audio_files)} archivos...")
    for filename in audio_files:
        path = os.path.join(input_folder, filename)
        feat_1024 = feature_extractor.get_24th_layer_features_averages(path, extract_with_padding=True)
        all_features_1024.append(feat_1024)
        filenames_order.append(filename)
        print(filename)

    data_matrix = np.array(all_features_1024) # Forma: (N_audios, 1024)

    # PASO 2: Calcular Entropía por dimensión
    print("Calculando entropía de las 1024 dimensiones...")
    entropies = calculate_entropy(data_matrix)

    # PASO 3: Seleccionar los índices de las 103 dimensiones con mayor entropía
    # argsort da los índices de menor a mayor, por eso usamos [-103:]
    best_indices = np.argsort(entropies)[-101:]
    
    print(f"Dimensiones seleccionadas basadas en alta entropía: {best_indices[:5]}... (total 101)")

    # PASO 4: Filtrar y Guardar
    for i, filename in enumerate(filenames_order):
        # Seleccionamos solo las 103 dimensiones ganadoras
        feat_101 = data_matrix[i, best_indices]
        
        # Forzamos formato float64 y forma (103,) como discutimos antes
        feat_101 = feat_101.astype(np.float64).flatten()

        new_filename = filename.replace("EN_", "ES_").replace(".wav", "_features.npy")
        output_path = os.path.join(output_folder, new_filename)
        
        np.save(output_path, feat_101)
        print(f"Procesado: {filename} -> {new_filename}") 
        
    print(f"¡Proceso completado! Archivos guardados en {output_folder}")    

def process_with_UMAP1(input_folder, output_folder):
    """
    1. Extrae características de 1024 dimensiones de todos los audios.
    2. Aplica PCA para reducir todas a 103 dimensiones.
    3. Guarda los archivos con el nombre correcto.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and f.startswith('EN_')]
    
    all_features_1024 = []
    filenames_order = []

    print(f"Paso 1: Extrayendo características originales de {len(audio_files)} archivos...")

    for filename in audio_files:
        path = os.path.join(input_folder, filename)
        # Extraer vector de 1024
        feat_1024 = feature_extractor.get_24th_layer_features_averages(path, extract_with_padding=True)
        
        all_features_1024.append(feat_1024)
        filenames_order.append(filename)

    # Convertir a matriz de numpy (Número de audios, 1024)
    data_matrix = np.array(all_features_1024)

    umap_reducer = UMAP(n_components=101, random_state=42)
    
    data_reduced  = umap_reducer.fit_transform(data_matrix)
   

    print(f"Paso 3: Guardando archivos .npy...")

    for i, filename in enumerate(filenames_order):
        # Obtener el vector de 103 dimensiones
        feat_101 = data_reduced[i] # Esto ya tiene forma (103,)

        # Cambiar nombre: ES_xxx_x.wav -> EN_xxx_x.npy
        new_filename = filename.replace("EN_", "ES_").replace(".wav", "_features.npy")
        output_path = os.path.join(output_folder, new_filename)
        feat_101 = np.array(feat_101, dtype=np.float64).flatten()
        np.save(output_path, feat_101)

    print(f"¡Hecho! Todos los archivos guardados en: {output_folder}")

def process_with_KBest1(input_folder, output_folder):
    """
    1. Extrae características de 1024 dimensiones de todos los audios.
    2. Aplica PCA para reducir todas a 103 dimensiones.
    3. Guarda los archivos con el nombre correcto.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and f.startswith('EN_')]
    
    all_features_1024 = []
    filenames_order = []

    print(f"Paso 1: Extrayendo características originales de {len(audio_files)} archivos...")

    for filename in audio_files:
        path = os.path.join(input_folder, filename)
        # Extraer vector de 1024
        feat_1024 = feature_extractor.get_24th_layer_features_averages(path, extract_with_padding=True)
        
        all_features_1024.append(feat_1024)
        filenames_order.append(filename)

    # Convertir a matriz de numpy (Número de audios, 1024)
    data_matrix = np.array(all_features_1024)

    # Supongamos que tienes etiquetas (si no, usa Varianza)
    selector = SelectKBest(k=101, score_func=lambda X, y: np.var(X, axis=0))
    
    data_reduced  = selector.fit_transform(data_matrix)  # si no tienes etiquetas
    
    print(f"Paso 3: Guardando archivos .npy...")

    for i, filename in enumerate(filenames_order):
        # Obtener el vector de 103 dimensiones
        feat_101 = data_reduced[i] # Esto ya tiene forma (103,)

        # Cambiar nombre: ES_xxx_x.wav -> EN_xxx_x.npy
        new_filename = filename.replace("EN_", "ES_").replace(".wav", "_features.npy")
        output_path = os.path.join(output_folder, new_filename)
        feat_101 = np.array(feat_101, dtype=np.float64).flatten()
        np.save(output_path, feat_101)

    print(f"¡Hecho! Todos los archivos guardados en: {output_folder}")

def process_with_Centroid1(input_folder, output_folder):
    """
    1. Extrae características de 1024 dimensiones de todos los audios.
    2. Aplica PCA para reducir todas a 103 dimensiones.
    3. Guarda los archivos con el nombre correcto.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and f.startswith('EN_')]
    
    all_features_1024 = []
    filenames_order = []

    print(f"Paso 1: Extrayendo características originales de {len(audio_files)} archivos...")

    for filename in audio_files:
        path = os.path.join(input_folder, filename)
        # Extraer vector de 1024
        feat_1024 = feature_extractor.get_24th_layer_features_averages(path, extract_with_padding=True)
        
        all_features_1024.append(feat_1024)
        filenames_order.append(filename)

    # Convertir a matriz de numpy (Número de audios, 1024)
    data_matrix = np.array(all_features_1024)

    kmeans = KMeans(n_clusters=101, random_state=42)
    
    data_reduced= kmeans.fit_transform(data_matrix)  # [1, 103]
   
  
    print(f"Paso 3: Guardando archivos .npy...")

    for i, filename in enumerate(filenames_order):
        # Obtener el vector de 103 dimensiones
        feat_101 = data_reduced[i] # Esto ya tiene forma (103,)

        # Cambiar nombre: ES_xxx_x.wav -> EN_xxx_x.npy
        new_filename = filename.replace("EN_", "ES_").replace(".wav", "_features.npy")
        output_path = os.path.join(output_folder, new_filename)
        feat_101 = np.array(feat_101, dtype=np.float64).flatten()
        np.save(output_path, feat_101)

    print(f"¡Hecho! Todos los archivos guardados en: {output_folder}")


def process_challenge_data_transf(input_folder, output_folder):
     """
     Procesa todos los audios de una carpeta, extrae características HuBERT,
     las reduce a 103 dimensiones y las guarda con el formato EN_xxx_x.npy
     """
     if not os.path.exists(output_folder):
        os.makedirs(output_folder)

     # Listar archivos .wav que empiecen con ES_
     audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and f.startswith('ES_')]
    
     print(f"Encontrados {len(audio_files)} archivos para procesar.")

     for filename in audio_files:
         path = os.path.join(input_folder, filename)
        
         # 1. Extraer características de la capa 24 (Dimensión 1024)
         # Usamos tu método existente
         feat_1024 = feature_extractor.extract_layer24(path,  add_padding=True, padding_sec=1.5)
         #print(feat_1024)
         # 2. Reducir a 103 dimensiones usando 'extract_winners'
         # Según las instrucciones, para el target de inglés se usan 103 dimensiones
         feat_103 = fs.extract_winners(feat_1024, "english")
        
         # 3. Asegurar que la forma sea (103,)
         feat_103 = np.array(feat_103, dtype=np.float64).flatten()
         #print( feat_103)
         
         if feat_103.shape != (103,):
            print(f"Error en dimensión para {filename}: {feat_103.shape}")
            continue

         # 4. Cambiar nombre de ES_xxx_x.wav a EN_xxx_x.npy
         new_filename = filename.replace("ES_", "EN_").replace(".wav", "_features.npy")
         output_path = os.path.join(output_folder, new_filename)

         
         # 5. Guardar como archivo numpy
         np.save(output_path, feat_103)
         print(f"Procesado: {filename} -> {new_filename}") 

def process_challenge_data_transf1(input_folder, output_folder):
     """
     Procesa todos los audios de una carpeta, extrae características HuBERT,
     las reduce a 101 dimensiones y las guarda con el formato ES_xxx_x.npy
     """
     if not os.path.exists(output_folder):
        os.makedirs(output_folder)

     # Listar archivos .wav que empiecen con ES_
     audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and f.startswith('EN_')]
    
     print(f"Encontrados {len(audio_files)} archivos para procesar.")

     for filename in audio_files:
         path = os.path.join(input_folder, filename)
        
         # 1. Extraer características de la capa 24 (Dimensión 1024)
         # Usamos tu método existente
         feat_1024 = feature_extractor.extract_layer24(path,  add_padding=True, padding_sec=1.5)
         #print(feat_1024)
         # 2. Reducir a 103 dimensiones usando 'extract_winners'
         # Según las instrucciones, para el target de inglés se usan 103 dimensiones
         feat_101 = fs.extract_winners(feat_1024, "spanish")
        
         # 3. Asegurar que la forma sea (103,)
         feat_101 = np.array(feat_101, dtype=np.float64).flatten()
         #print( feat_103)
         
         if feat_101.shape != (101,):
            print(f"Error en dimensión para {filename}: {feat_101.shape}")
            continue

         # 4. Cambiar nombre de ES_xxx_x.wav a EN_xxx_x.npy
         new_filename = filename.replace("EN_", "ES_").replace(".wav", "_features.npy")
         output_path = os.path.join(output_folder, new_filename)

         
         # 5. Guardar como archivo numpy
         np.save(output_path, feat_101)
         print(f"Procesado: {filename} -> {new_filename}") 
