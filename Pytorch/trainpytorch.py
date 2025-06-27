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
import os  # Para realizar operaciones del sistema operativo
import glob  # Para encontrar nombres de archivos que coincidan con un patrón
import wfdb  # Para trabajar con registros y anotaciones de datos fisiológicos
import sys  # Para manipular el intérprete de Python
import collections
import neurokit2 as nk  # Para procesar y analizar señales fisiológicas
import numpy as np  # Para operaciones numéricas y trabajo con arrays
import seaborn as sns
import tensorflow as tf  # Para trabajar con modelos de aprendizaje profundo
from tensorflow import keras  # API de alto nivel de TensorFlow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_signals import segmentSignals  # Función personalizada para segmentar señales
from sklearn.model_selection import train_test_split  # Para dividir datos en conjuntos de entrenamiento y prueba
from cnnpytorch import CNNModel  # Función personalizada para obtener un modelo CNN
from sklearn.model_selection import train_test_split, GridSearchCV, KFold  # Herramientas para validación cruzada y búsqueda de hiperparámetros
from scikeras.wrappers import KerasClassifier  # Envolver modelos Keras para usarlos con scikit-learn
from sklearn import metrics  # Para evaluar el rendimiento del modelo
import matplotlib.pyplot as plt  # Para generar gráficos
from sklearn.metrics import RocCurveDisplay, confusion_matrix, recall_score, f1_score  # Para mostrar curvas ROC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Parámetros constantes
FS = 500  # Frecuencia de muestreo
W_LEN = 256  # Longitud de la ventana para segmentar señales
W_LEN_1_4 = 256 // 4  # Un cuarto de la longitud de la ventana
W_LEN_3_4 = 3 * (256 // 4)  # Tres cuartos de la longitud de la ventana

def process_record(record_path, annotation_path):
    global latido

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
    segment = segmented_signals[1]  
    if not latido:
        sampling_rate = 500  
        time_axis = np.arange(len(segment)) / sampling_rate  

        r_peak_index = np.argmax(segment)  
        r_peak_value = segment[r_peak_index]  

        # Gráfica del latido y su pico R
        plt.figure(figsize=(10, 5))
        plt.plot(time_axis, segment, label="Latido Segmentado", color='red')  # Señal en rojo
        plt.scatter(time_axis[r_peak_index], r_peak_value, color='blue', zorder=5)  # Pico R en gris
        plt.text(time_axis[r_peak_index], r_peak_value, '  R', color='blue', fontsize=12) 
        plt.title("Latido segmentado y pico R")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Línea base
        plt.grid(True)
        plt.legend()
        plt.savefig('Pytorch/img/train/latidoconpico.png')
        plt.show()

        latido = True
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

import matplotlib.pyplot as plt

def plot_training_curves(train_loss, val_loss, train_acc, val_acc, modelname):
    # Obtener las métricas del historial
    epochs_range = range(len(train_loss))

    # Configurar el tamaño de la figura
    plt.figure(figsize=(12, 5))

    # Subplot 1: Curvas de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig(f"Pytorch/img/train/LossCurve{modelname}.png")
    plt.legend()

    # Subplot 2: Curvas de precisión
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(f"Pytorch/img/train/AccCurve{modelname}.png")
    plt.show()

# 📌 Verifica si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# 📌 Cargar datos
modelname = "FINAL2_500"
base_folder = "BBDD/ecg-id-database-1.0.0"
X = []  # Lista para las señales
y = []  # Lista para las etiquetas
latido = False  # Variable para controlar la gráfica del latido
# Procesar los datos
for person_id, person_folder in enumerate(sorted(glob.glob(os.path.join(base_folder, 'Person_*')))):
    print(f"Procesando persona: {person_id}")
    segments, labels = process_person(person_folder, person_id)  # Función que procesa señales
    X.extend(segments)
    y.extend(labels)

# Convertir datos a tensores de PyTorch
X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)  # Añadir canal
y = torch.tensor(np.array(y), dtype=torch.long)

# One-hot encoding
num_classes = 90
y = F.one_hot(y, num_classes=num_classes).float()

# Dividir en 80% entrenamiento + validación y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=torch.argmax(y, dim=1)
)

# Guardar conjunto de prueba para evaluaciones futuras
np.save(f"Pytorch/x_test{modelname}.npy", X_test.numpy())
np.save(f"Pytorch/y_test{modelname}.npy", y_test.numpy())

# 📌 Stratified KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_accuracies = []
best_model = None
best_accuracy = 0.0  
epochs = 500

best_train_losses = []
best_val_losses = []
best_train_accuracies = []
best_val_accuracies = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train, torch.argmax(y_train, dim=1))):
    print(f"Fold {fold}")

    # Dividir datos
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]


    # Crear DataLoaders
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    val_dataset = TensorDataset(X_val_fold, y_val_fold)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Instanciar el modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(seq_len=W_LEN, n_classes=90).to(device)  # Mover modelo a GPU si está disponible
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    early_stopping_patience = 10  # Número de épocas sin mejora antes de detener el entrenamiento
    best_val_loss = float("inf")
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    # Entrenamiento
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}")
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, torch.argmax(y_batch, dim=1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == torch.argmax(y_batch, dim=1)).sum().item()
            total_train += y_batch.size(0)

            train_loader_tqdm.set_postfix(loss=running_loss)

        # Guardar métricas de entrenamiento
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validación
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, torch.argmax(y_batch, dim=1))
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == torch.argmax(y_batch, dim=1)).sum().item()
                total_val += y_batch.size(0)

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f"Fold {fold + 1} - Epoch {epoch + 1} - Val Accuracy: {epoch_val_acc:.4f}")

        # Guardar mejor modelo
        if epoch_val_acc > best_accuracy:
            best_accuracy = epoch_val_acc
            best_model = model
            torch.save(model.state_dict(), f"Pytorch/{modelname}.pth")

            best_train_losses = train_losses.copy()
            best_val_losses = val_losses.copy()
            best_train_accuracies = train_accuracies.copy()
            best_val_accuracies = val_accuracies.copy()



plot_training_curves(
    train_loss=best_train_losses,
    val_loss=best_val_losses,
    train_acc=best_train_accuracies,
    val_acc=best_val_accuracies,
    modelname=modelname
)



# 📌 Matriz de Confusión
y_labels = torch.argmax(y, dim=1).cpu().numpy()
conf_matrix = confusion_matrix(y_labels, y_labels)


plt.figure(figsize=(15, 15))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Clase")
plt.ylabel("Clase")
plt.title("Matriz de Confusión de Frecuencias por Clase (Desbalanceo de Datos)")
plt.savefig(f"Pytorch/img/train/ConfMatrix{modelname}.png")
plt.show()

# 📌 Curva ROC para la clase 2 y 15 clases aleatorias
random_classes = np.random.choice(range(90), 15, replace=False).tolist()
classes_to_plot = [2] + random_classes  

plt.figure(figsize=(10, 7))

for class_id in classes_to_plot:
    fpr, tpr, _ = roc_curve(
        y_val_fold[:, class_id].cpu().numpy(),  # Cambia y_test por y_val_fold
        model(X_val_fold.to(device))[:, class_id].cpu().detach().numpy()
    )
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Clase {class_id} (AUC = {roc_auc:.2f})")


plt.plot([0, 1], [0, 1], 'k--', label="Chance Level (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC (One-vs-Rest) para 16 clases")
plt.legend()
plt.savefig(f"Pytorch/img/train/roc_curve{modelname}.png")  # Guarda la curva ROC como imagen
plt.show()

X_test_tensor = X_test.to(device)
y_test_tensor = y_test.to(device)

# 📌 Predicciones
best_model.eval()
with torch.no_grad():
    y_pred_probs = best_model(X_test_tensor)
    y_pred = torch.argmax(y_pred_probs, dim=1).cpu().numpy()
    y_true = torch.argmax(y_test_tensor, dim=1).cpu().numpy()

# 📌 Métricas
accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro')
recall_macro = recall_score(y_true, y_pred, average='macro')
f1_macro = f1_score(y_true, y_pred, average='macro')

precision_weighted = precision_score(y_true, y_pred, average='weighted')
recall_weighted = recall_score(y_true, y_pred, average='weighted')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

# 📌 Guardar métricas en un archivo txt
with open(f"Pytorch/metrics/metrics_summary{modelname}.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision (macro): {precision_macro:.4f}\n")
    f.write(f"Recall (macro): {recall_macro:.4f}\n")
    f.write(f"F1-score (macro): {f1_macro:.4f}\n")
    f.write(f"Precision (weighted): {precision_weighted:.4f}\n")
    f.write(f"Recall (weighted): {recall_weighted:.4f}\n")
    f.write(f"F1-score (weighted): {f1_weighted:.4f}\n")

# 📌 Guardar classification report completo
report = classification_report(y_true, y_pred, digits=4)
with open(f"Pytorch/metrics/classification_report{modelname}.txt", "w") as f:
    f.write(report)

# 📌 Matriz de Confusión
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(16, 16))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Conjunto de Prueba")
plt.savefig(f"Pytorch/metrics/confusion_matrix{modelname}.png")
plt.close()

# 📌 Guardar matriz en CSV
df_cm = pd.DataFrame(conf_matrix)
df_cm.to_csv(f"Pytorch/metrics/confusion_matrix.csv{modelname}", index=False)