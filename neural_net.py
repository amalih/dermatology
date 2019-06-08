

from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
numpy.random.seed(7)

data = np.loadtxt('dermatologyWOnan.data', delimiter=',')
#dermData.to_csv('dermData.csv')

target = data[:,34]  #provided your csv has header row, and the label column is named "Label"
data = data[:,:-1]

# create model
model = Sequential()
model.add(Dense(15, input_dim=35, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit model
model.fit(data, target, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(data, target)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
