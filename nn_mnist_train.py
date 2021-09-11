"""
    Создание нейрнной сети для распознавания рукописного текста
    с помощью обучающей выборки MNIST
    НС в данном проекте будет представлять собой сеть из полносвязных слоев, состоящих из:
    784 входных сигнала (Т.к. картинки библиотеки MNIST выложены с разрешением 28*28 пикселей)
    1 скрытый слой
    128 нейронов скрытого слоя
    10 нейронов выходного слоя
    функция активации ReLu
    Для выходного слоя softmax
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist              #библиотека базы выборок MNIST
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Стандартизация входных данных

x_train = x_train/255
x_test = x_test/255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)     # Приводит выходные данные у в вид вектора из 10 значений 0 и 1

# Формирование модели НС и вывод ее структуры в консоль

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

# Компиляция НС с опитимизацией по Adam и критерием - категориальная кросс-энтропия

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"]
              )

# Запуск процесса обучения: 80%-обучающая выборка, 20%-выборка валидации

model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

model.save("nn_mnist.h5")

#   Проверим НС на тестовой выборке
model.evaluate(x_test, y_test_cat)

# Проверка распознавания цифр

n = 0
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(f"Распознанная цифра: {np.argmax(res)}")

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()


#   Распознавание всей тестовой выборки

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])
print(y_test[:20])

# Выделение неверны вариантов

mask = pred == y_test
print(mask[:10])

x_false = x_test[~mask]
p_false = pred[~mask]

print(x_false.shape)

# Вывод 5 неверных результата

for i in range(5):
    print("Значение сети: "+str(y_test[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()

