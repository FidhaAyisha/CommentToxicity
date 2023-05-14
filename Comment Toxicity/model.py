from tenserflow.keras.models import sequential
from tenserflow.keras.layers import LSTM,Dropout,Birdirectional,Dense,Embedding
from matplotlib import pyplot as plt

model = Sequential()
model.add(Embedding(MAX_FEATURES+1,32))
model.add(Birdirectional(LSTM(32, activation= 'tanh')))

model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))

model.add(Dense(6,activation='sigmoid'))

model.compile(loss='BinaryCrosstropy', optimizer='Adam')

model.summary()

history = model.fit(train, epochs = 1, validation_data = val)

plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()