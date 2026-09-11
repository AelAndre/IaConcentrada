import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

__errors__ = []
__errors_val__ = []

X_train = pd.read_csv('X_train_seoul.csv').values.tolist()
Y_train = pd.read_csv('Y_train_seoul.csv').values.flatten().tolist()
X_val = pd.read_csv('X_val_seoul.csv').values.tolist()
Y_val = pd.read_csv('Y_val_seoul.csv').values.flatten().tolist()
X_test = pd.read_csv('X_test_seoul.csv').values.tolist()
Y_test = pd.read_csv('Y_test_seoul.csv').values.flatten().tolist()
print(f"train: {len(X_train)} filas, val: {len(X_val)} filas, test: {len(X_test)} filas")

for i in range(len(X_train)):
	X_train[i] = [1] + X_train[i]
for i in range(len(X_val)):
	X_val[i] = [1] + X_val[i]
for i in range(len(X_test)):
	X_test[i] = [1] + X_test[i]

random.seed(42)
params = [random.uniform(-0.01, 0.01) for _ in range(len(X_train[0]))]

def h(params, sample):
	acum = 0
	for i in range(len(params)):
		acum = acum + params[i]*sample[i]
	return acum

def GD(params, samples, y, alfa):
	temp = list(params)
	for j in range(len(params)):
		acum = 0
		for i in range(len(samples)):
			error = h(params,samples[i]) - y[i]
			acum = acum + error*samples[i][j]
		temp[j] = params[j] - alfa*(1/len(samples))*acum
	return temp

def scaling(samples, avgs=None, max_vals=None):
	samples = np.asarray(samples).T.tolist()
	calcular = avgs is None
	if calcular:
		avgs = [0]*len(samples)
		max_vals = [0]*len(samples)
	for i in range(1, len(samples)):
		if calcular:
			acum = 0
			for j in range(len(samples[i])):
				acum = acum + samples[i][j]
			avgs[i] = acum/len(samples[i])
			max_vals[i] = max(samples[i])
		for j in range(len(samples[i])):
			samples[i][j] = (samples[i][j] - avgs[i])/max_vals[i]
	samples = np.asarray(samples).T.tolist()
	return samples, avgs, max_vals

X_train_s, avgs, max_vals = scaling(X_train)
X_val_s, _, _ = scaling(X_val, avgs, max_vals)
X_test_s, _, _ = scaling(X_test, avgs, max_vals)

def show_errors(params, samples, y, error_list=None):
	global __errors__
	if error_list is None:
		error_list = __errors__
	error_acum = 0
	for i in range(len(samples)):
		hyp = h(params, samples[i])
		error = hyp - y[i]
		error_acum = error_acum + error**2
	mean_error_param = error_acum/len(samples)
	error_list.append(mean_error_param)
	return mean_error_param

alfa = 0.05
max_epochs = 300
epochs = 0
while True:
	oldparams = list(params)
	params = GD(params, X_train_s, Y_train, alfa)
	show_errors(params, X_train_s, Y_train)
	show_errors(params, X_val_s, Y_val, __errors_val__)
	epochs = epochs + 1
	if (oldparams == params or epochs == max_epochs):
		break
print("epochs:", epochs)
print("params finales:", params)

plt.plot(__errors__, label="Entrenamiento")
plt.plot(__errors_val__, label="Validación")
plt.xlabel("Épocas"); plt.ylabel("Error (MSE)"); plt.legend()
plt.show()

def predict(samples, params):
	return [h(params, s) for s in samples]

def mse_manual(y_real, y_pred):
	error_acum = 0
	for i in range(len(y_real)):
		error_acum = error_acum + (y_pred[i]-y_real[i])**2
	return error_acum/len(y_real)

y_pred_val = predict(X_val_s, params)
y_pred_test = predict(X_test_s, params)
print("MSE validación:", mse_manual(Y_val, y_pred_val))
print("MSE test:", mse_manual(Y_test, y_pred_test))