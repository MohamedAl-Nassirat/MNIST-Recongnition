import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1) #minimizes the range from 0-1

model = tf.keras.models.Sequential() # Feed forward model
model.add(tf.keras.layers.Flatten()) # flattens the layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #128 layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #Second hidden layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #Output layer, 10 nodes to account for 0-9 digits


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Loss = Caluclation for error, measuring accuracy

model.fit(x_train, y_train, epochs=3) # Fit the model, 

val_loss, val_acc = model.evaluate(x_test, y_test) 
print(val_loss)
print(val_acc)




model.save('epic_num_reader.model') #loads the model
new_model = tf.keras.models.load_model('epic_num_reader.model') #loading new model back


predictions = new_model.predict(x_test)
print(np.argmax(predictions[3]))
plt.imshow(x_test[3],cmap=plt.cm.binary)
plt.show()