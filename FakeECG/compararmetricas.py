import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, ks_2samp
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
from tqdm import tqdm

# Parámetros
label = 27  # La persona real es label + 1
dir_path = f"FakeECG/TestPersona_{label+1}"

# Cargar datos
X_train = np.load("FakeECG/X_train500.npy")
y_train = np.load("FakeECG/y_train500.npy")

# Convierte one-hot a índices (clases reales de 0 a 89)
y_train_indices = np.argmax(y_train, axis=1)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("Valores únicos en y_train:", np.unique(y_train_indices))
print(f"Cantidad de señales reales para label {label}:", np.sum(y_train_indices == label))

# Filtrar las señales reales de la persona deseada
X_real = X_train[y_train_indices == label].squeeze()  # (N, 256)
X_fake = np.load(f"{dir_path}/X_test_fake_person_{label+1}.npy").squeeze()  # (N, 256)

# Verificar las formas
print("Forma de X_real:", X_real.shape)
print("Forma de X_fake:", X_fake.shape)

# Verifica si hay señales reales y sintéticas
if X_real.shape[0] == 0 or X_fake.shape[0] == 0:
    raise ValueError(f"No se encontraron señales para la persona {label+1}. Verifica los archivos y las rutas.")

# Inicializar listas de métricas
mse_list, mae_list, dtw_list, pearson_list, cos_sim_list = [], [], [], [], []

# Comparar señales reales vs sintéticas
for real, fake in tqdm(zip(X_real, X_fake), total=len(X_real)):
    mse_list.append(mean_squared_error(real, fake))
    mae_list.append(mean_absolute_error(real, fake))
    
    dtw_dist, _ = fastdtw(real, fake)
    dtw_list.append(dtw_dist)

    pearson_corr, _ = pearsonr(real, fake)
    pearson_list.append(pearson_corr)

    cos_sim = 1 - cosine(real, fake)
    cos_sim_list.append(cos_sim)

# Estadística de distribuciones (Kolmogorov-Smirnov)
ks_stat, ks_pval = ks_2samp(X_real.flatten(), X_fake.flatten())

# Mostrar resultados
print("\n🔎 Métricas promedio:")
print(f"MSE medio: {np.mean(mse_list):.4f}")
print(f"MAE medio: {np.mean(mae_list):.4f}")
print(f"DTW medio: {np.mean(dtw_list):.4f}")
print(f"Pearson medio: {np.mean(pearson_list):.4f}")
print(f"Cosine similarity media: {np.mean(cos_sim_list):.4f}")

print("\n📊 Test de Kolmogorov-Smirnov:")
print(f"KS statistic: {ks_stat:.4f}, p-value: {ks_pval:.4f}")
