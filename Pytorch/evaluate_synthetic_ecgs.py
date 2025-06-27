import os
import subprocess
import torch
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from cnnpytorch import CNNModel


# CONFIGURACIÓN
W_LEN = 256
N_CLASSES = 90
N_ITER = 10
modelname = "FINAL_500"
generate_script = "FakeECG/generate_all_ECG_2.py"

# Cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(seq_len=W_LEN, n_classes=N_CLASSES).to(device)
model.load_state_dict(torch.load(f"Pytorch/{modelname}.pth", map_location=device))
model.eval()

# Rutas
base_folder = "BBDD"

# Inicializar métricas
all_true = []
all_pred = []
errores_acumulados = np.zeros(N_CLASSES, dtype=int)
total_por_clase = np.zeros(N_CLASSES, dtype=int)

# BUCLE PRINCIPAL
for iteration in range(1, N_ITER + 1):
    print(f"\n🔁 Iteración {iteration}/{N_ITER}: Generando datos sintéticos...")

    result = subprocess.run(["python", generate_script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] Error en generación: {result.stderr}")
        continue
    print(result.stdout)

    # Predicción para cada persona
    for i in range(1, N_CLASSES + 1):
        persona = f"{i:02d}"
        record_path = os.path.join(base_folder, f"Person_{persona}", "rec_1")

        if not os.path.exists(record_path + ".hea"):
            print(f"[!] Archivo no encontrado: {record_path}.hea")
            continue

        try:
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal[:, 0][:W_LEN]
            synthetic_beat = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(synthetic_beat)
                predicted_class = torch.argmax(prediction, dim=1).item()

            true_label = i - 1
            all_true.append(true_label)
            all_pred.append(predicted_class)

            total_por_clase[true_label] += 1
            if predicted_class != true_label:
                errores_acumulados[true_label] += 1

        except Exception as e:
            print(f"[!] Error procesando Persona {persona}: {e}")

# ---------------------
# MÉTRICAS Y GRÁFICAS
# ---------------------

accuracy = sum([t == p for t, p in zip(all_true, all_pred)]) / len(all_true)
print(f"\n✅ Precisión total promedio tras {N_ITER} iteraciones: {accuracy:.2%}")

errores_prom = errores_acumulados / total_por_clase

# Matriz de confusión
cm = confusion_matrix(all_true, all_pred, labels=range(N_CLASSES))
fig, ax = plt.subplots(figsize=(12, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{i+1}" for i in range(N_CLASSES)])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Matriz de Confusión Global")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("Pytorch/img/test/MatrizConfAll.png")

plt.show()

# Error por clase
plt.figure(figsize=(12, 5))
plt.bar(range(1, N_CLASSES + 1), errores_prom)
plt.xlabel("Persona (clase)")
plt.ylabel("Error medio (fallos / muestras)")
plt.title("Error promedio por clase tras 10 iteraciones")
plt.grid(True)
plt.tight_layout()
plt.savefig("Pytorch/img/test/ErrorPorClase5.png")
plt.show()

# Top clases con más errores
top_fails = np.argsort(-errores_prom)[:10]
print("\n🔍 Clases con mayor tasa de error promedio:")
for idx in top_fails:
    print(f"Persona {idx+1:02d}: {errores_prom[idx]:.2%} de error (fallos: {errores_acumulados[idx]}, total: {total_por_clase[idx]})")
