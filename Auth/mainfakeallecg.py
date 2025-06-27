import torch
import numpy as np
import wfdb
from cnnpytorch import CNNModel
import os

# Parámetros
FS = 500
W_LEN = 256
N_CLASSES = 90
modelname = "FINAL_500"

# Cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(seq_len=W_LEN, n_classes=N_CLASSES).to(device)
model.load_state_dict(torch.load(f"Pytorch/{modelname}.pth", map_location=device))
model.eval()

# Ruta base
base_folder = "BBDD/ecg-id-database-1.0.0"

# Inicializar contadores
correct = 0
total = 0

# Iterar sobre personas
for i in range(1, N_CLASSES + 1):
    Persona = f"{i:02d}"  # Añadir 0 delante si es <10
    synthetic_ecg = f"BBDD/Person_{Persona}/rec_1"

    try:
        record = wfdb.rdrecord(synthetic_ecg)
        signal = record.p_signal[:, 0]  # Solo primer canal

        # Preprocesamiento (ajustar si es necesario)
        signal = signal[:W_LEN]  # Asegurarse de que tenga la longitud correcta
        synthetic_beat = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Predicción
        with torch.no_grad():
            predictions = model(synthetic_beat)
            predicted_class = torch.argmax(predictions, dim=1).item()

        # Comparar con la clase real
        if predicted_class == i - 1:  # Índices 0 a 89
            correct += 1
        total += 1

        print(f"Persona {Persona}: Predicho {predicted_class+1} - {'Correcto' if predicted_class == i - 1 else 'Incorrecto'}")

    except Exception as e:
        print(f"No se pudo procesar la Persona {Persona}: {e}")

# Calcular precisión
accuracy = correct / total if total > 0 else 0
print(f"\nPrecisión total del modelo con latidos sintéticos: {accuracy:.2%}")
