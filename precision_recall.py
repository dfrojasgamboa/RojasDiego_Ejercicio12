import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.metrics
from sklearn.preprocessing import StandardScaler # z = (x - u) / s. Standardize features by removing the mean and scaling to unit variance
from sklearn.model_selection import train_test_split # Split arrays or matrices into random train and test subsets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # dimensionality reduction technique retaining as much information as possible.

# DATA SET

# lee los numeros
numeros = skdata.load_digits()

# lee los labels
target = numeros['target']

# lee las imagenes
imagenes = numeros['images']

# cuenta el numero de imagenes total
n_imagenes = len(target)

# para poder correr PCA debemos "aplanar las imagenes"
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))


# Split en train/test
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.5) # train_size

# todo lo que es diferente de 1 queda marcado como 0
y_train[y_train!=1]=0
y_test[y_test!=1]=0


# Reescalado de los datos
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



# Encuentro los autovalores y autovectores de las imagenes marcadas como 1.
numero = 1
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)

# pueden ser complejos por baja precision numerica, asi que los paso a reales
valores = np.real(valores)
vectores = np.real(vectores)

# reordeno de mayor a menor
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

# encuentro las imagenes en el espacio de los autovectores
x_test_transform = x_test @ vectores
x_train_transform = x_train @ vectores


# inicializo el clasificador
linear = LinearDiscriminantAnalysis()

def F1( precision, recall ):
    return 2 * (precision * recall) / (precision + recall)
    
    
n_components = 10

linear.fit(x_train_transform[:,:n_components], y_train) # training data and parameters according to the number of components.
pred_prob = linear.predict_proba(x_test_transform[:,:n_components]) # Predict class labels for samples in x_train_transform for train data
precision, recall, threshold = sklearn.metrics.precision_recall_curve( y_test, probas_pred=pred_prob[:,1])

F1_array = F1( precision, recall)
max_F1 = np.max( F1_array )


# Encuentro los autovalores y autovectores de las imagenes marcadas como 1.
numero != 1
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)

# pueden ser complejos por baja precision numerica, asi que los paso a reales
valores = np.real(valores)
vectores = np.real(vectores)

# reordeno de mayor a menor
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

# encuentro las imagenes en el espacio de los autovectores
x_test_transform = x_test @ vectores
x_train_transform = x_train @ vectores

n_components = 10

linear.fit(x_train_transform[:,:n_components], y_train) # training data and parameters according to the number of components.
pred_prob = linear.predict_proba(x_test_transform[:,:n_components]) # Predict class labels for samples in x_train_transform for train data
print(np.shape(pred_prob[:,1]))
print(np.shape(y_test[:-1]))
precision, recall, threshold = sklearn.metrics.precision_recall_curve( y_test, probas_pred=pred_prob[:,1])

F1_array = F1( precision, recall)
max_F1 = np.max( F1_array )

# hago la grafica
plt.figure()
plt.subplot(1,2,1)
plt.plot(threshold, F1_array[:-1])
plt.xlabel('Probability')
plt.ylabel('F1')

plt.subplot(1,2,2)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.subplots_adjust(wspace=0.5)

plt.savefig('F1_prec_recall.png')