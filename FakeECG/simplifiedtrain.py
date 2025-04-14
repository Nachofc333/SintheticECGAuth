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
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

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
num_epochs = 80
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



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=torch.argmax(y, dim=1))

persona_objetivo = 1  # Cambia este número a la persona que quieres
indices = (y_train == persona_objetivo).nonzero(as_tuple=True)[0]

X_train_filtrado = X_train[indices]
y_train_filtrado = y_train[indices]


# 📌 Dimensiones del modelo
input_dim = X_train.shape[-1]  # Longitud de la señal ECG
latent_dim = 16  # Dimensión del espacio latente



# 📌 Instanciar el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cvae = ConditionalVAE(in_channels=1, num_classes=num_classes, latent_dim=latent_dim, seq_length=W_LEN).to(device)
optimizer = optim.Adam(cvae.parameters(), lr=0.001)

# 📌 Función de pérdida (Reconstrucción + KL Divergence)
def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# 📌 Entrenamiento del CVAE

train_dataset = TensorDataset(X_train_filtrado, y_train_filtrado)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

"""train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)"""

for epoch in range(num_epochs):
    cvae.train()
    total_loss = 0
    for x, labels in train_loader:
        x, labels = x.to(device), labels.to(device)
        optimizer.zero_grad()
        
        recon_x, x_hat, mu, logvar = cvae(x, labels)  
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 📌 Generación de señales falsas
def generate_ecg(model, label):
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)  # Generar muestra aleatoria
        label_tensor = F.one_hot(torch.tensor([label]), num_classes=num_classes).float().to(device)  # One-hot encoding
        z = torch.cat([z, label_tensor], dim=1)  # Concatenar con la etiqueta
        fake_ecg = model.decode(z).cpu().numpy()  # Pasar por el decoder
    return fake_ecg

def save_synthetic_ecg(fake_signal, label):
    """
    Guarda el ECG sintético en formato .dat, .hea y .atr.
    """
    output_dir = "BBDD/Person_00"
    os.makedirs(output_dir, exist_ok=True)  # Crear la carpeta si no existe
    
    filename = f"rec_1"
    file_path = os.path.join(output_dir, filename)

    # 🔹 Asegurar que los datos sean float64
    p_signal = np.array(fake_signal, dtype=np.float64).reshape(-1, 1)  # WFDB requiere 2D array

    # 🔹 Guardar el ECG sintético con metadatos básicos
    wfdb.wrsamp(
        file_path,
        fs=500,  # Frecuencia de muestreo (ajústala según tu dataset)
        units=["mV"],  # Unidad de medida
        sig_name=["ECG_lead1"],  # Nombre de la señal
        p_signal=p_signal
    )

    print(f"ECG sintético guardado en {file_path}.dat y {file_path}.hea")

ecg_real = X_train[0].cpu().numpy()
# 📌 Generar un ECG falso para una persona específica
fake_signal = generate_ecg(cvae, label=0)
save_synthetic_ecg(fake_signal, label=0)

print(f"Forma de ecg_real antes: {np.shape(ecg_real)}")
print(f"Forma de fake_signal antes: {np.shape(fake_signal)}")
print(f"Tipo de ecg_real: {type(ecg_real)}, Tipo de fake_signal: {type(fake_signal)}")
print(f"Dimensión de ecg_real: {ecg_real.ndim}, Dimensión de fake_signal: {fake_signal.ndim}")

print(f"Min y max de ECG real: {ecg_real.min()}, {ecg_real.max()}")
print(f"Min y max de ECG generado: {fake_signal.min()}, {fake_signal.max()}")

# Obtener min y max del ECG real
min_real, max_real = ecg_real.min(), ecg_real.max()

# Desnormalizar el ECG generado
#fake_signal = (fake_signal + 1) * (max_real - min_real) / 2 + min_real

# Verificar los nuevos valores
print(f"Min y max de ECG generado después de desnormalizar: {fake_signal.min()}, {fake_signal.max()}")


plt.plot(fake_signal[0][0])  # Graficar ECG generado
plt.title("ECG Sintético Generado por CVAE")
plt.savefig("ECGGENERADO.png")
plt.show()

# 📌 Graficar ambos ECGs
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(ecg_real[0])  
plt.title("ECG Real - Persona 1")

plt.subplot(1, 2, 2)
plt.plot(fake_signal[0][0])  
plt.title("ECG Generado - Persona 1")

plt.savefig("comparacion_ecg.png")
plt.show()
