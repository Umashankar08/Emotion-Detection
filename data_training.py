import os  
import numpy as np 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

# Initialization
is_init = False
size = -1

label = []
dictionary = {}
c = 0

# Load data from .npy files
for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
        if not(is_init):
            is_init = True 
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]]*size).reshape(-1,1)
        else:
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c = c + 1

# Convert labels to integers
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# One-hot encode labels
y = to_categorical(y)

# Shuffle data
X_new = X.copy()
y_new = y.copy()
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i, idx in enumerate(cnt): 
    X_new[i] = X[idx]
    y_new[i] = y[idx]

# Define the model
ip = Input(shape=(X.shape[1],))
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)

# Compile the model
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model
model.fit(X_new, y_new, epochs=50)

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
