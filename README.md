# Рублёв Владимир Сергеевич ББМО-02-23

CleverHans — это известная библиотека, предназначенная для изучения и защиты моделей машинного обучения от враждебных атак. Разработанная командой Google Brain, она стала одной из первых платформ, ориентированных на тестирование устойчивости моделей к атакующим примерам.

Foolbox — инструмент для проверки надежности моделей машинного обучения путем проведения враждебных атак. Она предлагает удобные средства для создания атакующих примеров, которые способны вызывать ошибки в работе модели.

Ссылка на работу в google colab: https://colab.research.google.com/drive/1fs_oiO0eeFvmd3GzIxaHtU0_ev0YzaIr#scrollTo=c9cUxOYcRfJ-

# Практика 1: Установка окружения и настройка фреймворков для анализа защищенности ИИ

CleverHans — это ещё одна популярная библиотека для работы с враждебными атаками, разработанная для проведения тестирования и защиты моделей от атакующих примеров. CleverHans был одной из первых библиотек, ориентированных на adversarial attacks, и изначально был создан командой Google Brain.

Foolbox — это библиотека для проведения атак на модели машинного обучения с целью тестирования их устойчивости к adversarial attacks (враждебным атакам). Она предоставляет удобные методы для создания атакующих примеров, которые могут вызывать некорректное поведение модели.

# Установка необходимых библиотек

!pip install foolbox
!pip install cleverhans

![image](https://github.com/vladimirrublev/AZSII-1-/blob/main/screenshot/1.png)

# Проверка установленных библиотек

```
# Импортируем необходимые библиотеки
import tensorflow as tf
import torch
import foolbox
import cleverhans

# Проверяем версии установленных библиотек
print(f"Версия TensorFlow: {tf.__version__}")
print(f"Версия PyTorch: {torch.__version__}")
print(f"Версия Foolbox: {foolbox.__version__}")
print(f"Версия CleverHans: {cleverhans.__version__}")

```
![image](https://github.com/vladimirrublev/AZSII-1-/blob/main/screenshot/22.png)

# Загрузка и обучение модели

```
# Импорт необходимых модулей из TensorFlow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Загрузка и подготовка данных MNIST
(train_X, train_y), (test_X, test_y) = mnist.load_data()  # Загрузка набора данных
train_X = train_X.astype('float32') / 255.0  # Нормализация тренировочных изображений
test_X = test_X.astype('float32') / 255.0    # Нормализация тестовых изображений

# Преобразование меток в формат one-hot
train_y = to_categorical(train_y, num_classes=10)
test_y = to_categorical(test_y, num_classes=10)

# Создание архитектуры нейронной сети
model = Sequential([
    Flatten(input_shape=(28, 28)),             # Разворачивание изображений в вектор
    Dense(units=128, activation='relu'),      # Скрытый слой с 128 нейронами
    Dense(units=10, activation='softmax')     # Выходной слой для классификации
])

# Компиляция модели
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Обучение модели на тренировочных данных
model.fit(train_X, train_y, epochs=5, batch_size=32)

# Оценка точности модели на тестовых данных
loss, accuracy = model.evaluate(test_X, test_y)
print(f"Accuracy on test data: {accuracy:.2f}")

```

![image](https://github.com/vladimirrublev/AZSII-1-/blob/main/screenshot/33.png)

# Сохранение модели

```

#Сохранение модели в рабочую директорию google colab
model.save('mnist_model.h5')
#скачивание файла на локальный компьютер сразу
from google.colab import files
files.download('mnist_model.h5')

```

![image](https://github.com/vladimirrublev/AZSII-1-/blob/main/screenshot/44.png)
