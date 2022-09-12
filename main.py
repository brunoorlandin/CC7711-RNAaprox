import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

print('Carregando Arquivo de teste')
arquivo = np.load('teste5.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])

minErrors = []

data = {}

for i in range(10):
    regr = MLPRegressor(hidden_layer_sizes=(1000,150),
                        max_iter=100000,
                        activation='logistic', #{'identity', 'logistic', 'tanh', 'relu'},
                        solver='adam',
                        learning_rate = 'adaptive',
                        n_iter_no_change=50)
    print('Treinando RNA %d' %(i+1))
    regr = regr.fit(x,y)

    print('Preditor %d' %(i+1))
    y_est = regr.predict(x)

    minErrors.append(regr.best_loss_)

    minError = regr.best_loss_

    data[minError] = [regr.loss_curve_.copy(), y_est.copy()]

minError = min(data.keys())

curve = data[minError][0]
y_est = data[minError][1]

plt.figure(figsize=[14,7])

#plot curso original
plt.subplot(1,3,1)
plt.plot(x,y)

#plot aprendizagem
plt.subplot(1,3,2)
plt.plot(curve)

#plot regressor
plt.subplot(1,3,3)
plt.plot(x,y,linewidth=1,color='yellow')
plt.plot(x,y_est,linewidth=2)

plt.show()

print("Média do erro: " + str(round(np.average(minErrors),2)))
print("Desvio padrão do erro: " + str(round(np.std(minErrors),5)))
