# generate_submission.py
import os
import sys
import numpy as np
import torch
import torchaudio
import soundfile as sf
from sklearn.decomposition import PCA
import joblib
import zipfile

# ==============================
# CONFIGURACIÓN (AJUSTA SEGÚN TU ENTORNO)
# ==============================
data_dir = "../data"
pred_dir = os.path.join(data_dir, "spanish_test")
feature_dir = feature_directory = os.path.join(data_dir, "features2")
INPUT_AUDIO_DIR = pred_dir        # ← Cambia si tus audios están en otra carpeta
OUTPUT_FEATURES_DIR = feature_dir
PCA_PATH = "pca_hubert_l_103.pkl"       # PCA preentrenado o a entrenar
BUNDLE = "hubert_l"                     # o "wav2vec_l"
N_FEATURES = 103
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# CLASE FeatureExtractor (adaptada para producción)
# ==============================
class FeatureExtractor:
    def __init__(self, bundle=BUNDLE, n_features=N_FEATURES, pca_path=PCA_PATH, device=DEVICE):
        self.bundle = self._get_bundle(bundle)
        self.model = self.bundle.get_model().to(device)
        self.device = device
        self.n_features = n_features
        self.pca_path = pca_path
        self.pca = self._load_pca()

    def _get_bundle(self, bundle):
        if bundle == "hubert_l":
            return torchaudio.pipelines.HUBERT_LARGE
        elif bundle == "wav2vec_l":
            return torchaudio.pipelines.WAV2VEC2_LARGE
        else:
            raise ValueError("Solo 'hubert_l' o 'wav2vec_l'.")

    def _load_pca(self):
        if not os.path.exists(self.pca_path):
            print(f"[ERROR] PCA no encontrado: {self.pca_path}")
            print("👉 Por favor, entrena el PCA primero o coloca un archivo válido.")
            sys.exit(1)
        pca = joblib.load(self.pca_path)
        print(f"[INFO] PCA cargado: {pca.n_components} componentes, "
              f"{pca.explained_variance_ratio_.sum():.1%} varianza explicada.")
        return pca

    def _load_audio(self, path):
        data, sr = sf.read(path)
        data = data.astype(np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1)
        waveform = torch.from_numpy(data).unsqueeze(0).to(self.device)
        if sr != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.bundle.sample_rate
            )
        return waveform

    def _extract_layer24(self, waveform, add_padding=True, padding_sec=1.5):
        with torch.inference_mode():
            features, _ = self.model.extract_features(waveform)
        layer24 = features[23]  # [1, T, 1024]

        if add_padding:
            frames_per_sec = self.bundle.sample_rate / 320  # 50 fps
            f_remove = int(padding_sec * frames_per_sec)
            layer24 = layer24[:, f_remove:-f_remove, :]

        return layer24

    def extract_features_103(self, audio_path):
        """Devuelve un np.ndarray de forma (103,)"""
        waveform = self._load_audio(audio_path)
        layer24 = self._extract_layer24(waveform, add_padding=True)

        # Promedio en tiempo → [1024]
        avg_1024 = layer24.squeeze(0).mean(dim=0).cpu().numpy()  # shape: (1024,)

        # PCA → [103]
        avg_103 = self.pca.transform(avg_1024.reshape(1, -1)).flatten()  # shape: (103,)

        assert avg_103.shape == (103,), f"Error: forma {avg_103.shape}, esperado (103,)"
        return avg_103


# ==============================
# FUNCIÓN PRINCIPAL: GENERAR SUBMISSION
# ==============================
def main():
    # Verificar que exista INPUT_AUDIO_DIR
    if not os.path.exists(INPUT_AUDIO_DIR):
        print(f"[ERROR] Carpeta no encontrada: {INPUT_AUDIO_DIR}")
        print("👉 Coloca todos los audios (ES_xxx_x.wav) en esta carpeta.")
        sys.exit(1)

    # Crear carpeta de salida
    os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True)

    # Inicializar extractor
    try:
        extractor = FeatureExtractor()
    except Exception as e:
        print(f"[FATAL] Error al cargar el extractor: {e}")
        sys.exit(1)

    # Listar todos los archivos .wav que empiecen con "ES_"
    audio_files = [
        f for f in os.listdir(INPUT_AUDIO_DIR)
        if f.endswith(".wav") and f.startswith("ES_")
    ]

    if not audio_files:
        print(f"[ERROR] No se encontraron archivos 'ES_xxx_x.wav' en {INPUT_AUDIO_DIR}")
        sys.exit(1)

    print(f"[INFO] Procesando {len(audio_files)} archivos...")

    # Procesar cada audio
    for i, audio_file in enumerate(sorted(audio_files), 1):
        input_path = os.path.join(INPUT_AUDIO_DIR, audio_file)
        # Cambiar nombre: ES_xxx_x.wav → EN_xxx_x.npy
        output_name = audio_file.replace("ES_", "EN_").replace(".wav", ".npy")
        output_path = os.path.join(OUTPUT_FEATURES_DIR, output_name)

        try:
            features = extractor.extract_features_103(input_path)
            # Guardar como .npy (automáticamente guarda como ndarray 1D)
            np.save(output_path, features)
            print(f"[{i}/{len(audio_files)}] ✅ {audio_file} → {output_name} (forma: {features.shape})")
        except Exception as e:
            print(f"[{i}/{len(audio_files)}] ❌ Error en {audio_file}: {e}")
            sys.exit(1)

    # Crear ZIP listo para CodaBench
    zip_name = "submission.zip"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUTPUT_FEATURES_DIR):
            for file in files:
                if file.endswith(".npy"):
                    full_path = os.path.join(root, file)
                    # Guardar SIN subcarpetas → archivos directos en el ZIP
                    zf.write(full_path, arcname=file)  # ← clave: arcname=file (sin ruta)
                    print(f"[ZIP] Añadido: {file}")

    print(f"\n🎉 ¡Listo! Archivo ZIP creado: {zip_name}")
    print("✅ Este archivo cumple con los requisitos de CodaBench:")
    print("   - Todos los .npy tienen forma (103,)")
    print("   - Nombres: EN_xxx_x.npy")
    print("   - ZIP contiene SOLO los archivos .npy (sin carpetas)")
    print("\n📤 Sube 'submission.zip' a CodaBench.")


if __name__ == "__main__":
    main()