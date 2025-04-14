# FakeECGAuth

Work in progress


### BBDD: 
Base de datos con los registros de los latidos de 90 personas 
## -------------------------------------------------------------------
## Elementos autenticador

### Pytorch/cnnpytorch: 
Este archivo contiene el modelo CNN utilizado para entrenar el autenticador
### Pytorch/trainpytorch: 
Este archivo contiene el entrenamiento del modelo utilizando `cnnpytorch.py` y su generacion en formato .pth
### segment_signals: 
Este archivo contiene el codigo para segmentar y procesar las señales de los ECG
### Pytorch/mainpytorch: 
Archivo que contiene el codigo encargado de realizar las prediciones seleccionando un latido real
### Pytorch/mainpytorch: 
Archivo que contiene el codigo encargado de realizar las prediciones seleccionando un latido sintético generado
### Pytorch/model.pth:
Modelo entrenado para predecir a quien pertenece el latido
### Pytorch/multiplefakeecg:
Metricas utilizando multiples ecgs sinteticos y los latidos reales para comparar

## -----------------------------------------------------------------

## Elementos generador

### FakeECG/CVAE.py
Autoencoder variacional condiconal entrenado con todos los ecg reales. Recibe la clase como condicion para entrenar y generar electrocardiogramas de dicha clase 
### FakeECG/generate_ecg.py
Codigo para generar un latido sintetico de una persona en concreto indicada
### FakeECG/train_cvae.py
Este archivo se utiliza para entrenar la CVAE y generar el modelo en formato .pth

