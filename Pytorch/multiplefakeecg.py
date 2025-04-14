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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ----------- Configuración ----------
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
modelname = "M500"
Persona = "3"
real_class = int(Persona) - 1  # Persona 3 → clase 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- Cargar modelo ----------
model = CNNModel(seq_len=256, n_classes=90).to(device)
model.load_state_dict(torch.load(f"Pytorch/{modelname}.pth", map_location=device))
model.eval()

# ----------- Cargar datos sintéticos ----------
X_fake = np.load(f"FakeECG/TestPersona_{Persona}/X_test_fake_person_{Persona}.npy")
y_fake = np.load(f"FakeECG/TestPersona_{Persona}/y_test_fake_person_{Persona}.npy")

# Pasar a tensores
X_fake_tensor = torch.tensor(X_fake, dtype=torch.float32).to(device)

# ----------- Predicciones sintéticas ----------
with torch.no_grad():
    y_prob_fake = model(X_fake_tensor)
    y_pred_fake = torch.argmax(y_prob_fake, dim=1).cpu().numpy()

# ----------- Reporte para datos sintéticos ----------
print(f"\n📊 Evaluación de ECGs Sintéticos - Persona {Persona} (Clase {real_class})")
print("Reporte de clasificación completo:")
print(classification_report(y_fake, y_pred_fake, digits=4, zero_division=0))

# Extraer métricas específicas de la persona (sintéticos)
report_fake = classification_report(y_fake, y_pred_fake, output_dict=True, zero_division=0)
fake_metrics = report_fake[str(real_class)]

print(f"\n🔎 Métricas específicas para ECGs Sintéticos de Persona {Persona}:")
print(f"Precision: {fake_metrics['precision']:.4f}")
print(f"Recall:    {fake_metrics['recall']:.4f}")
print(f"F1-score:  {fake_metrics['f1-score']:.4f}")

# ============================================================================


# ----------- Cargar datos reales ----------
X_test = torch.tensor(np.load(f"Pytorch/x_test{modelname}.npy"))  # Ya guardado previamente
y_test = torch.tensor(np.load(f"Pytorch/y_test{modelname}.npy"))  # One-hot codificado

X_test_tensor = X_test.to(device)
y_test_tensor = y_test.to(device)

# ----------- Predicciones reales ----------
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_tensor)
    y_pred = torch.argmax(y_pred_probs, dim=1).cpu().numpy()
    y_true = torch.argmax(y_test_tensor, dim=1).cpu().numpy()

# ----------- Reporte para datos reales ----------
print(f"\n📊 Evaluación de ECGs Reales - Todos los sujetos")
print("Reporte de clasificación completo:")
print(classification_report(y_true, y_pred, digits=4, zero_division=0))

# ----------- Extraer métricas específicas para la misma persona (reales) ----------
report_real = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

if str(real_class) in report_real:
    real_metrics = report_real[str(real_class)]
    print(f"\n🔎 Métricas específicas para ECGs Reales de Persona {Persona}:")
    print(f"Precision: {real_metrics['precision']:.4f}")
    print(f"Recall:    {real_metrics['recall']:.4f}")
    print(f"F1-score:  {real_metrics['f1-score']:.4f}")
else:
    print(f"\n❌ La clase {real_class} no está presente en los datos reales.")

# (Opcional) Puedes guardar también las métricas comparadas en un archivo si lo deseas
