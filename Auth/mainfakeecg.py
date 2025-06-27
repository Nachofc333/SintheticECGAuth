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

# Constantes para el procesamiento de las señales de ECG
FS = 500  # Frecuencia de muestreo
W_LEN = 256  # Longitud de la ventana
W_LEN_1_4 = 256 // 4  # Un cuarto de la longitud de la ventana
W_LEN_3_4 = 3 * (256 // 4)  # Tres cuartos de la longitud de la ventana

# Configuración del modelo
W_LEN = 256  # Longitud de la ventana de entrada
N_CLASSES = 90  # Número de clases
modelname = "FINAL_500"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(seq_len=W_LEN, n_classes=N_CLASSES).to(device)

# Cargar los pesos del modelo entrenado
model.load_state_dict(torch.load(f"Auth/{modelname}.pth", map_location=device))
model.eval()

X_test = torch.tensor(np.load(f"Auth/x_test{modelname}.npy"), dtype=torch.float32).to(device)
y_test = np.load(f"Auth/y_test{modelname}.npy")  # No en tensor para manipular índices
y_train = np.load("Auth/y_train.npy")  # No en tensor para manipular índices

# Cargar los datos (asumiendo que ya están preprocesados)
# Carpeta base donde se encuentran los datos de ECG
base_folder = "BBDD/ecg-id-database-1.0.0"
#---------------------------------------------------------------------------------------------------------
Persona = "30" # SI ES < 10 PONER 0 DELANTE
#---------------------------------------------------------------------------------------------------------
synthetic_ecg = f"BBDD/Person_{Persona}/rec_1"

record = wfdb.rdrecord(synthetic_ecg)
signal = record.p_signal[:, 0]  # Usar solo el primer canal

synthetic_beat = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# Realizar la predicción
with torch.no_grad():
    predictions = model(synthetic_beat)
    predicted_class = torch.argmax(predictions, dim=1).item()

print(f"El latido pertenece a la persona: {Persona}")
print(f"PREDICCIÓN: El latido pertenece a la persona: {predicted_class+1}")

# -------------------- Seleccionar Latido Real -------------------- 
indices = np.where(y_test == predicted_class)[0]  # Buscar índices con la persona predicha
print(f"Clases únicas en y_test: {np.unique(y_test)}")
print(f"Clases únicas en y_train: {np.unique(y_train)}")

print(indices)


person_folder = f"BBDD/ecg-id-database-1.0.0/Person_{predicted_class+1:02d}"  # Ajustar formato XX
record_paths = glob.glob(os.path.join(person_folder, "*.hea"))
#record_paths = glob.glob(os.path.join(person_folder, "*.dat"))  # Buscar archivos de ECG

person_folder2 = f"BBDD/ecg-id-database-1.0.0/Person_{Persona}"  # Ajustar formato XX
record_paths2 = glob.glob(os.path.join(person_folder2, "*.hea"))
# Graficar la señal
# Cargar conjunto de prueba
X_test = torch.tensor(np.load("x_test.npy"), dtype=torch.float32).to(device)
y_test = torch.tensor(np.load("y_test.npy"), dtype=torch.long).to(device)

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
plt.savefig("Auth/img/test/ROCmain.png")
plt.show()

if record_paths:
    real_record_path = record_paths[0][:-4]  # Eliminar la extensión .dat para obtener la base
    real_record = wfdb.rdrecord(real_record_path)
    annotation_path = real_record_path  # Mismo nombre para el archivo de anotaciones

    print(f"Procesando persona: {person_folder}")  # Mostrar progreso
    real_beat_segments = process_record(real_record_path, annotation_path)
    print(len(real_beat_segments))
    real_beat = real_beat_segments[0]  # Seleccionar el primer latido segmentado
    
    real_record_path2 = record_paths2[0][:-4]  # Eliminar la extensión .dat para obtener la base
    real_record2 = wfdb.rdrecord(real_record_path2)
    annotation_path2 = real_record_path2  # Mismo nombre para el archivo de anotaciones

    print(f"Procesando persona: {person_folder2}")  # Mostrar progreso
    real_beat_segments2 = process_record(real_record_path2, annotation_path2)
    print(len(real_beat_segments2))
    real_beat2 = real_beat_segments2[7]  # Seleccionar el primer latido segmentado

    # **Graficar ambos latidos**
    plt.figure(figsize=(15, 4))

    # ECG Sintético
    plt.subplot(1, 3, 1)
    plt.plot(synthetic_beat.cpu().numpy()[0, 0])
    plt.title(f"ECG Sintético (Persona {Persona})")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.grid()

    # ECG Real predicho
    plt.subplot(1, 3, 2)
    plt.plot(real_beat)
    plt.title(f"ECG Real predicho (Persona {predicted_class+1})")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(real_beat2)
    plt.title(f"ECG Real (Persona {Persona})")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"Auth/img/test/comparacion_ecg{Persona}.png")
    plt.show()
else:
    print(f"No se encontraron registros reales para la persona {predicted_class}.")