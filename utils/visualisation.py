import matplotlib.pyplot as plt
import torch
import librosa
import librosa.display
import numpy as np
import pandas as pd
import seaborn as sns
from config import LABEL_MAP

def plot_waveform(waveform, sr, title="Waveform"):
    """Plot a waveform."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    
    if waveform.ndim > 1:
        waveform = waveform.squeeze()
    
    plt.figure(figsize=(10, 4))
    plt.title(title)
    librosa.display.waveshow(waveform, sr=sr)
    plt.tight_layout()
    return plt.gcf()

def plot_spectrogram(spectrogram, title="Mel Spectrogram"):
    """Plot a spectrogram."""
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.numpy()
    
    if spectrogram.ndim > 2:
        spectrogram = spectrogram.squeeze()
    
    plt.figure(figsize=(10, 4))
    plt.title(title)
    librosa.display.specshow(
        spectrogram, 
        x_axis='time', 
        y_axis='mel',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    return plt.gcf()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot a confusion matrix."""
    if class_names is None:
        class_names = list(LABEL_MAP.keys())
        
    cm = pd.DataFrame(
        np.zeros((len(class_names), len(class_names))), 
        index=class_names, 
        columns=class_names
    )
    
    for i in range(len(y_true)):
        cm.iloc[y_true[i], y_pred[i]] += 1
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt.gcf()

def visualize_embeddings(embeddings, labels, method='tsne'):
    """Visualize embeddings using dimensionality reduction."""
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    
    # Convert embeddings to numpy for sklearn
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    # Reduce dimensions
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Map numerical labels to class names
    label_names = [list(LABEL_MAP.keys())[list(LABEL_MAP.values()).index(l)] 
                  for l in labels]
    
    # Plot
    plt.figure(figsize=(10, 8))
    for label in set(label_names):
        idx = [i for i, l in enumerate(label_names) if l == label]
        plt.scatter(
            reduced_embeddings[idx, 0], 
            reduced_embeddings[idx, 1], 
            label=label
        )
    
    plt.legend()
    plt.title(f'{method.upper()} Visualization of Embeddings')
    plt.tight_layout()
    return plt.gcf()