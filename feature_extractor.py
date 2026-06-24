import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
from transformers import AutoProcessor, AutoModel

class FeatureExtractor:
    """
    Uses pytorch to extract transformation layers from an audio file.
    Utilizes either the HuBERT Large, Wav2Vec2.0 Large, or WavLM Large models
    bundle: String : 'hubert_l' or 'wav2vec_l' or 'wavlm_l' expected
    """
    def __init__(self, bundle='hubert_l'):
        torch.random.manual_seed(0)  # Sets the same random weights everytime the model is run.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = self.get_bundle(bundle)
        self.model = self.bundle.get_model().to(self.device)
        self.print_info()

    def get_bundle(self, bundle):
        if bundle == "hubert_l": return torchaudio.pipelines.HUBERT_LARGE    
        if bundle == "wav2vec_l": return torchaudio.pipelines.WAV2VEC2_LARGE
        if bundle == "wavlm_l": return torchaudio.pipelines.WAVLM_LARGE
        if bundle == "wavlm_300": return torchaudio.pipelines.WAV2VEC2_XLSR_300M
        if bundle == "wavlm_l60K": return torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
        if bundle == "wavlm_l60K_10": return torchaudio.pipelines.WAV2VEC2_LARGE_LV60K_10M
        if bundle == "wavlm_robust_60_utt": return torchaudio.pipelines.WAV2VEC2_LARGE_ROBUST
        if bundle == "wavlm_robust": return torchaudio.pipelines.WAV2VEC2_LARGE_LV60K_UTTERANCE_CMN
        if bundle == "wavlm_unis_sat": return torchaudio.pipelines.UNISPEECH_SAT_LARGE
        if bundle == "wavlm_l60K_960H": return torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
        if bundle == "wavlm_robust_utt": return torchaudio.pipelines.WAV2VEC2_LARGE_ROBUST_UTTERANCE_CMN
        print(f"bundle name {bundle} not recognized: 'hubert_l' or 'wav2vec_l' or 'wavlm_l' expected.")
        sys.exit()

    def print_info(self):
        print(f"torch version: {torch.__version__}")
        print(f"torch audio version: {torchaudio.__version__}")
        print(f"device: {self.device}")
        print(f"Sample Rate: {self.bundle.sample_rate}")
        print(f"model class: {self.model.__class__}")
        
    def _load_audio(self, path):
        """Carga audio con soundfile (mono, 16kHz)."""
        data, sr = sf.read(path)
        if data.ndim == 2:
            data = data.mean(axis=1)
        # Re-muestrear a 16kHz (requerido por WavLM)
        if sr != 16000:
            from scipy.signal import resample
            data = resample(data, int(len(data) * 16000 / sr))
        return data.astype(np.float32)

    def extract_layer24(self, audio_path, add_padding=True, padding_sec=1.5):
        """Extrae features de la capa 24 (índice 23) con padding opcional."""
        self.model_name = "facebook/wav2vec2-xls-r-300m"   #facebook/wav2vec2-large-lv60, facebook/wav2vec2-xls-r-1b, microsoft/unispeech-sat-large, microsoft/beats-large, facebook/data2vec2-audio-large facebook/wav2vec2-xls-r-2b facebook/w2v-bert-2.0 facebook/hubert-conformer-large
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, output_hidden_states=True).to(self.device)
        
        audio = self._load_audio(audio_path)
        
        # Añadir silencio al inicio y final si se pide
        if add_padding:
            pad_samples = int(padding_sec * 16000)
            padded_audio = np.zeros(len(audio) + 2 * pad_samples)
            padded_audio[pad_samples:-pad_samples] = audio
            audio = padded_audio
        
        # Preprocesar con Hugging Face
        inputs = self.processor(
            audio, sampling_rate=16000,
            return_tensors="pt", padding=True
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Obtener capa 24 (índice 23, porque 0 es embedding)
        layer = outputs.hidden_states[23]  # [B, T, 1024]
        
        # Eliminar padding (si se añadió)
        #if add_padding:
         #   frames_per_sec = 16000 // 320  # 50 fps (como en HuBERT)
         #   f_remove = int(padding_sec * frames_per_sec)
         #   layer24 = layer24[:, f_remove:-f_remove, :]

        if add_padding:
            return self.get_features_averages_from_tl([self.remove_padding_from_feats(layer)])[0]
        else:
            return self.get_features_averages_from_tl([layer])[0]
            
        #return layer24

    def get_transformation_layers(self, path, add_padding=False, padding_length_in_seconds=1.5, plot_layers=False):
        """
        Passes an audio file to a self-supervised machine learning model.
        :param path: path of the audio file : String
        :param add_padding: adds silence padding to beginning and end of audio clip if true.
        :param padding_length_in_seconds: length of silence padding to add in seconds.
        :param plot_layers: Will visualize the transformation layers if true.
        :return: A list of 24 3d tensors representing the 24 transformation layers
        Tensor Dimensions:
        1st = number of audio files processed at once
        2nd = number of frames per audio file (One frame for every 20ms)
        3rd = number of features extracted per frame (typically 1024)
        """
        if not os.path.exists(path):
            print(f"ERROR: {path} is not a valid file path")
            return
        #waveform, sample_rate = torchaudio.load(path)
        #waveform, sample_rate = torchaudio.load(path, backend="soundfile")
        #data, sample_rate = sf.read(path)
        #waveform = torch.from_numpy(data).float()
        #if len(waveform.shape) > 1:
         #waveform = waveform.mean(dim=1, keepdim=True)
        #else:
         #waveform = waveform.unsqueeze(0) # Añade la dimensión de canal [1, samples]
        waveform, sample_rate = librosa.load(path,sr=self.bundle.sample_rate,mono=True)
        waveform = torch.from_numpy(waveform).unsqueeze(0).float()  
        
        waveform = waveform.to(self.device)
        if sample_rate != self.bundle.sample_rate:
         waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)
        
        # Add silence padding if specified - stabilizes feature extraction at the edges of the audio clip
        if add_padding:
            padding_length_in_samples = int(padding_length_in_seconds * self.bundle.sample_rate)
            silence_padding = torch.zeros((1, padding_length_in_samples)).to(self.device)
            waveform = torch.cat((silence_padding, waveform, silence_padding), dim=1)
        
        with torch.inference_mode():  # Disables gradient computation and back propagation.
            features, _ = self.model.extract_features(waveform)
        if plot_layers: self.plot_layers(features)
        return features

    def get_features_averages_from_tl(self, transformation_layers):
        """
        Calculates feature averages of each transformation layer given a list of transformation layers.
        :param transformation_layers: a list of transformation layers.
        :return: A list of size len(transformation_layers). Each element in the
        list is a numpy array of size 1024 representing the average values of the 1024
        features the SSL models calculate per frame of the audio clip.
        """
        all_tl_averages = []
        for layer in transformation_layers:
            layer = torch.squeeze(layer, dim=0) # get rid of first dimension
            tl_averages = [] # will hold averages for this one transformation layer
            num_features = layer.shape[1] # columns
            num_frames = layer.shape[0] # rows
            for feature_idx in range(num_features):
                feature_sum = 0
                for frame_idx in range(num_frames):
                    feature_sum += layer[frame_idx, feature_idx].item()
                tl_averages.append(feature_sum / num_frames)
            all_tl_averages.append(np.array(tl_averages))
        return all_tl_averages


    def remove_padding_from_feats(self, features, feats_to_remove=75):
        """
        Function to remove padding silence features from beginning and end of audio clip.
        The padding was added in order to get better feature extraction from the SSL models.
        :param features: 3d tensor representing transformation layer features.
        :param feats_to_remove: number of features to remove from beginning and end. Change if frame rate changes (now based on 1500/20ms).
        :return: 3d tensor representing transformation layer features without silence features.
        """
        return features[:, feats_to_remove:-feats_to_remove, :]

    def get_24th_layer_features_averages(self, file_path, extract_with_padding=False):
        """
        We found the 24th layer had the highest correlation when compared to human judgements.
        This method takes in a file path and returns the feature averages from the 24th layer only.
        :param file_path: path where the audio clip is stored.
        :param extract_with_padding: whether to add padding silence when extracting features.
        :return: a 1d numpy array of size 1024 that represent the average value of the features
        calculated in the 24th layer.
        """
        self.model = self.bundle.get_model().to(self.device)
        transformation_layers = self.get_transformation_layers(file_path, add_padding=extract_with_padding)
        layer = transformation_layers[23] # use only the 24th layer
        if extract_with_padding:
            return self.get_features_averages_from_tl([self.remove_padding_from_feats(layer)])[0]
        else:
            return self.get_features_averages_from_tl([layer])[0]

    def plot_layers(self, features):
        """
        Visualizes transformation layers from an audio file.
        :param features: List of 3d tensor objects representing transformation layers.
        :return: None
        """
        fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
        for i, feats in enumerate(features):
            ax[i].imshow(feats[0].cpu(), interpolation="nearest")
            ax[i].set_title(f"Feature from transformer layer {i + 1}")
            ax[i].set_xlabel("Feature dimension")
            ax[i].set_ylabel("Frame (time-axis)")
        plt.tight_layout()
        plt.show()

    