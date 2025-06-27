import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset
from cnnpytorch import CNNModel
import os

# --- CONFIGURACIÓN ---
MODEL_PATH = "Auth/FINAL_500.pth"
N_CLASSES = 90
SEQ_LEN = 256 # Cambia si tu longitud de secuencia es diferente
BATCH_SIZE = 32

# --- CARGAR DATOS ---
# Asegúrate de que X_test y y_test estén ya procesados (normalizados, en tensores, etc.)
X_test = np.load("Auth/x_testFINAL_500.npy")  # o usa tu propio código para cargar
y_test = np.load("Auth/y_testFINAL_500.npy")  # one-hot encoded

# Si y_test está one-hot encoded, convierte a etiquetas
if y_test.ndim > 1:
    y_test_labels = np.argmax(y_test, axis=1)
else:
    y_test_labels = y_test

# --- PREPROCESAMIENTO A TENSORES ---
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_labels, dtype=torch.long)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- CARGAR MODELO ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(seq_len=SEQ_LEN, n_classes=N_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- EVALUACIÓN ---
all_probs = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        labels = y_batch.cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)

all_probs = np.concatenate(all_probs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# --- ONE-HOT PARA ROC ---
y_true_onehot = label_binarize(all_labels, classes=np.arange(N_CLASSES))

# --- CALCULAR AUC POR CLASE ---
aucs = []
for i in range(N_CLASSES):
    try:
        auc = roc_auc_score(y_true_onehot[:, i], all_probs[:, i])
    except ValueError:
        auc = np.nan
    aucs.append(auc)

# --- RESULTADOS ---
print("\nAUC-ROC por clase:")
for i, auc in enumerate(aucs):
    if np.isnan(auc):
        print(f"Clase {i:2d}: sin muestras en test")
    else:
        print(f"Clase {i:2d}: AUC = {auc:.4f}")

worst_class = np.nanargmin(aucs)
print(f"\n🚨 Peor clase: {worst_class} con AUC = {aucs[worst_class]:.4f}")