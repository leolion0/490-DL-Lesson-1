import pandas
from keras.models import Sequential
import keras.optimizers
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()


dataset = pd.read_csv("breastcancer.csv")
# print(dataset)
X = dataset.drop(columns=['diagnosis'])
X = X.iloc[:, :-1].values
X = sc.fit_transform(X)
Y = dataset['diagnosis']
Y = Y.str.replace('B', '0')
Y = Y.str.replace('M', '1')


# print(X)
# print(Y)
# exit(-1)
nAdam = keras.optimizers.Adam(lr=.0001)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.33, random_state=87)
np.random.seed(155)
my_first_nn = Sequential()  # create model
my_first_nn.add(Dense(8, input_dim=31, activation='relu'))  # hidden layer
my_first_nn.add(Dense(8, activation='relu')  )# hidden layer
my_first_nn.add(Dense(8, activation='relu') ) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid'))  # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='nAdam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=500,
                                     initial_epoch=0)
print("-----------------------------ayyy b ------------------------------------------------------\n\n\n")
# print(my_first_nn.summary())
print(my_first_nn.metrics_names)
print(my_first_nn.evaluate(X_test, Y_test))
