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
from CVAEpruebas import ConditionalVAE
from VAE import EstandarVAE
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

# Parámetros
num_classes = 90
seq_length = 256
batch_size = 16
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
    segmentos = sorted(glob.glob(os.path.join(person_folder, '*.hea')))

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
def plot_training_curves(train_loss, val_loss, modelname):
    epochs_range = range(1, len(train_loss))

    plt.figure(figsize=(6, 5))
    plt.plot(epochs_range, train_loss[1:], label='Training Loss')
    plt.plot(epochs_range, val_loss[1:], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(f"FakeECG/img/train/LossCurve{modelname}.png")
    plt.show()


def plot_cvae_losses(recon_losses, kl_losses, modelname):
    epochs_range = range(len(recon_losses))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, recon_losses, label='Reconstruction Loss')
    plt.plot(epochs_range, kl_losses, label='KL Divergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('CVAE Losses')
    plt.legend()
    plt.savefig(f"FakeECG/img/train/CVAE_Losses_{modelname}.png")
    plt.show()


# 📌 Verifica si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# 📌 Cargar datos
base_folder = "BBDD/ecg-id-database-1.0.0"
modelname = "FINAL5_300"
num_classes = 90
num_epochs = 300
latent_dim = 32
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
np.save(f"FakeECG/X_train{modelname}.npy", X_train.cpu().numpy())  # Guardar en formato NumPy
np.save(f"FakeECG/y_train{modelname}.npy", y_train.cpu().numpy())  # Guardar en formato NumPy

# 📌 Dataset y DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_test, y_test)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 📌 Instanciar modelo
cvae = ConditionalVAE(in_channels=1, num_classes=num_classes, latent_dim=latent_dim, seq_length=seq_length).to(device)
optimizer = optim.Adam(cvae.parameters(), lr=0.001)

# 📌 Función de pérdida
def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

train_losses = []
val_losses = []
val_accuracies = []
recon_losses = []
kl_losses = []


# 📌 Entrenamiento
for epoch in range(num_epochs):
    cvae.train()
    total_loss = 0
    for x, labels in train_loader:
        x, labels = x.to(device), labels.to(device)
        optimizer.zero_grad()
        
        recon_x, _, mu, logvar = cvae(x, labels)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss

        loss.backward()
        recon_losses.append(recon_loss.item() / x.size(0))
        kl_losses.append(kl_loss.item() / x.size(0))
        optimizer.step()
        
        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # 📌 Evaluación en validación
    cvae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val, labels_val in val_loader:
            x_val, labels_val = x_val.to(device), labels_val.to(device)
            recon_x_val, _, mu_val, logvar_val = cvae(x_val, labels_val)
            val_batch_loss = loss_function(recon_x_val, x_val, mu_val, logvar_val)
            val_loss += val_batch_loss.item()
    
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

# 📌 Guardar modelo entrenado
torch.save(cvae.state_dict(), f"FakeECG/{modelname}.pth")
print(f"Modelo guardado exitosamente en 'FakeECG/{modelname}.pth'")


# 📈 Dibujar curvas de entrenamiento

plot_training_curves(train_losses, val_losses, modelname=modelname)
plot_cvae_losses(recon_losses, kl_losses, modelname)
