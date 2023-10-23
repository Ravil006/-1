import tensorflow as tf 
from tensorflow import keras

#Загрузка и предоработка данных, загрузка датасета
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

#нормализуем пиксели в диапозоне от 0 до 1
x_train, x_test = x_train/255.0, x_test/255.0

#шаг 2. Оперделение арх-ры НС
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense( units: 128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
])

#шаг 3. Компиляция
model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])

#шаг 4. Обучение модели
model.fit(x_train, y_train, epochs=10)

#шаг 5. Оценка модели на тест данных
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Точность на тестовых данных:{test_acc}')