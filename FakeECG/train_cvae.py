import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from tqdm import tqdm 
# Importar las bibliotecas necesarias
import wfdb  # Para trabajar con registros y anotaciones de datos fisiológicos
import sys  # Para manipular el intérprete de Python
import collections
import neurokit2 as nk  # Para procesar y analizar señales fisiológicas
from segment_signals import segmentSignals  # Función personalizada para segmentar señales
from sklearn.model_selection import train_test_split  # Para dividir datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split, GridSearchCV, KFold  # Herramientas para validación cruzada y búsqueda de hiperparámetros
from sklearn import metrics  # Para evaluar el rendimiento del modelo
import matplotlib.pyplot as plt  # Para generar gráficos
from sklearn.metrics import RocCurveDisplay, confusion_matrix, recall_score, f1_score  # Para mostrar curvas ROC
from CVAE import ConditionalVAE
from VAE import EstandarVAE
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

# Parámetros
num_classes = 90
latent_dim = 16
seq_length = 256
batch_size = 32
num_epochs = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parámetros constantes
FS = 500  # Frecuencia de muestreo
W_LEN = 256  # Longitud de la ventana para segmentar señales
W_LEN_1_4 = 256 // 4  # Un cuarto de la longitud de la ventana
W_LEN_3_4 = 3 * (256 // 4)  # Tres cuartos de la longitud de la ventana

def process_record(record_path, annotation_path):
    # Leer el registro desde el archivo
    record = wfdb.rdrecord(record_path)
    # Leer las anotaciones desde el archivo
    annotation = wfdb.rdann(annotation_path, 'atr')

    # Obtener la señal y la frecuencia de muestreo
    signal = record.p_signal[:, 0]  # Solo el primer canal
    sampling_rate = record.fs

    # Procesar la señal con NeuroKit para limpiarla
    signals, info = nk.ecg_process(signal, sampling_rate=sampling_rate)
    signal = signals["ECG_Clean"]  # Señal limpia
    r_peaks_annot = info["ECG_R_Peaks"]  # Posiciones de los picos R

    # Segmentar latidos de la señal
    segmented_signals, refined_r_peaks = segmentSignals(signal, r_peaks_annot)
    return segmented_signals

def process_person(person_folder, person_id):
    # Inicializar listas para almacenar segmentos y etiquetas
    all_segments = []
    all_labels = []
    segmentos = sorted(glob.glob(os.path.join(person_folder, '*.hea')))[:2]

    # Iterar sobre cada archivo en la carpeta de la persona
    for record_path in segmentos:
        base = record_path[:-4]  # Eliminar la extensión del archivo
        annotation_path = base  # Ruta de las anotaciones

        # Procesar el archivo y segmentar los latidos
        segments = process_record(base, annotation_path)
        all_segments.extend(segments)  # Agregar segmentos
        all_labels.extend([person_id] * len(segments))  # Agregar etiquetas correspondientes

    # Convertir las listas en arrays de NumPy
    return np.array(all_segments), np.array(all_labels)

# Graficar las curvas de pérdida y precisión
def plot_training_curves(history):
    # Obtener las métricas del historial
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(1, len(history.history['loss']) + 1)

    # Configurar el tamaño de la figura
    plt.figure(figsize=(12, 5))

    # Subplot 1: Curvas de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Subplot 2: Curvas de precisión
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.show()

# 📌 Verifica si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# 📌 Cargar datos
base_folder = "BBDD/ecg-id-database-1.0.0"
num_classes = 90
num_epochs = 500
batch_size = 32
X = []  # Lista para las señales
y = []  # Lista para las etiquetas

# Procesar los datos
for person_id, person_folder in enumerate(sorted(glob.glob(os.path.join(base_folder, 'Person_*')))):
    print(f"Procesando persona: {person_id}")
    segments, labels = process_person(person_folder, person_id)  # Función que procesa señales
    X.extend(segments)
    y.extend(labels)


# Convertir datos a tensores de PyTorch
X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)  # Añadir canal
y = torch.tensor(np.array(y), dtype=torch.long)
y = F.one_hot(y, num_classes).float()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar X_train en un archivo npy
np.save("FakeECG/X_train500.npy", X_train.cpu().numpy())  # Guardar en formato NumPy
np.save("FakeECG/y_train500.npy", y_train.cpu().numpy())  # Guardar en formato NumPy

# 📌 Dataset y DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 📌 Instanciar modelo
cvae = ConditionalVAE(in_channels=1, num_classes=num_classes, latent_dim=latent_dim, seq_length=seq_length).to(device)
optimizer = optim.Adam(cvae.parameters(), lr=0.001)

# 📌 Función de pérdida
def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# 📌 Entrenamiento
for epoch in range(num_epochs):
    cvae.train()
    total_loss = 0
    for x, labels in train_loader:
        x, labels = x.to(device), labels.to(device)
        optimizer.zero_grad()
        
        recon_x, _, mu, logvar = cvae(x, labels)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 📌 Guardar modelo entrenado
torch.save(cvae.state_dict(), "FakeECG/model500.pth")
print("Modelo guardado exitosamente en 'cvae_model.pth'")