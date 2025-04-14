import torch
import torch.nn as nn
import torch.optim as optim
import neurokit2 as nk  # Librería para procesamiento de señales fisiológicas
import numpy as np  # Librería para manejo de arrays numéricos
import os  # Para operaciones con el sistema de archivos
import glob  # Para buscar archivos con patrones específicos
import wfdb  # Librería para manejo de datos en formato WFDB
import matplotlib.pyplot as plt  # Para graficar datos
from segment_signals import segmentSignals  # Función personalizada para segmentar señales
from cnnpytorch import CNNModel
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
# Constantes para el procesamiento de las señales de ECG
FS = 500  # Frecuencia de muestreo
W_LEN = 256  # Longitud de la ventana
W_LEN_1_4 = 256 // 4  # Un cuarto de la longitud de la ventana
W_LEN_3_4 = 3 * (256 // 4)  # Tres cuartos de la longitud de la ventana


def process_record(record_path, annotation_path):
    """
    Procesa un registro de ECG y devuelve los segmentos del latido.

    Args:
        record_path (str): Ruta al archivo del registro.
        annotation_path (str): Ruta al archivo de anotaciones.

    Returns:
        np.array: Señales segmentadas de latidos.
    """
    # Leer el registro y las anotaciones del archivo
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(annotation_path, 'atr')

    # Obtener la señal y la frecuencia de muestreo
    signal = record.p_signal[:, 0]  # Usar solo el primer canal
    sampling_rate = record.fs

    # Procesar la señal para limpiar y extraer picos R
    signals, info = nk.ecg_process(signal, sampling_rate=sampling_rate)
    signal = signals["ECG_Clean"]  # Señal de ECG limpia
    r_peaks_annot = info["ECG_R_Peaks"]  # Picos R detectados

    # Segmentar los latidos a partir de los picos R
    segmented_signals, refined_r_peaks = segmentSignals(signal, r_peaks_annot)
    return segmented_signals

def process_person(person_folder, person_id):
    """
    Procesa todos los registros de una persona y devuelve los segmentos y etiquetas.

    Args:
        person_folder (str): Carpeta con los registros de la persona.
        person_id (int): ID de la persona.

    Returns:
        tuple: Arrays de segmentos y etiquetas.
    """
    all_segments = []  # Lista para almacenar segmentos
    all_labels = []  # Lista para almacenar etiquetas

    # Buscar todos los archivos de cabecera (.hea) en la carpeta
    for record_path in glob.glob(os.path.join(person_folder, '*.hea')):
        base = record_path[:-4]  # Eliminar la extensión para obtener la base del nombre
        annotation_path = base  # Asumir que el archivo de anotaciones tiene el mismo nombre

        # Procesar el archivo y segmentar los latidos
        segments = process_record(base, annotation_path)
        all_segments.extend(segments)  # Agregar segmentos a la lista
        all_labels.extend([person_id] * len(segments))  # Agregar etiquetas correspondientes
 
    return np.array(all_segments), np.array(all_labels)

# Configuración del modelo
W_LEN = 256  # Longitud de la ventana de entrada
N_CLASSES = 90  # Número de clases
modelname = "M500"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(seq_len=W_LEN, n_classes=N_CLASSES).to(device)

# Cargar los pesos del modelo entrenado
model.load_state_dict(torch.load(f"Pytorch/{modelname}.pth", map_location=device))
model.eval()

# Cargar los datos (asumiendo que ya están preprocesados)
# Carpeta base donde se encuentran los datos de ECG
base_folder = "BBDD/ecg-id-database-1.0.0"
x = []  # Lista para almacenar segmentos de latidos
y = []  # Lista para almacenar etiquetas

# Iterar sobre las carpetas de cada persona en la base de datos
for person_id, person_folder in enumerate(sorted(glob.glob(os.path.join(base_folder, 'Person_*')))):
    print(f"Procesando persona: {person_folder}")  # Mostrar progreso
    segments, labels = process_person(person_folder, person_id)  # Procesar registros de la persona
    x.extend(segments)  # Agregar segmentos procesados a la lista
    y.extend(labels)  # Agregar etiquetas correspondientes

x = np.array(x)
y = np.array(y)

x = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)  # Añadir dimensión de canal

etiqueta = 6200
new_beat = x[etiqueta].unsqueeze(0)  # Agregar dimensión de batch

y_true = y[etiqueta]
print("Shape de new_beat antes de la predicción:", new_beat.shape)

# Realizar la predicción
with torch.no_grad():
    predictions = model(new_beat)
    predicted_class = torch.argmax(predictions, dim=1).item()

print(f"La etiqueta real del latido en la posición {etiqueta} es: {y_true}")
print(f"PREDICCIÓN: El latido pertenece a la persona: {predicted_class}")

# Graficar la señal

plt.plot(new_beat.cpu().numpy()[0, 0])
plt.title(f"Latido segmentado (Predicción: Persona {predicted_class})")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid()
plt.savefig("latido.png")  # Guarda la curva ROC como imagen
plt.show()

# Cargar el mejor modelo
model.load_state_dict(torch.load(f"Pytorch/{modelname}.pth"))
model.to(device)  # Mover el modelo a GPU si está disponible
model.eval()

# Cargar conjunto de prueba
X_test = torch.tensor(np.load(f"Pytorch/x_test{modelname}.npy"), dtype=torch.float32).to(device)
y_test = torch.tensor(np.load(f"Pytorch/y_test{modelname}.npy"), dtype=torch.long).to(device)

with torch.no_grad():
    y_prob = model(X_test).cpu().numpy()

# Obtener etiquetas reales
y_true = np.argmax(y_test.cpu().numpy(), axis=1)

# Generar curva ROC

n_classes = 90
y_test_bin = label_binarize(y_true, classes=range(n_classes))

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Graficar curva ROC
plt.figure(figsize=(15, 10))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Clase {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC Multiclase")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("ROCmain.png")
plt.show()