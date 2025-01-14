## Описание

Этот проект предназначен для генерации подписей к изображениям с последующим переводом на русский язык. Он использует
модель `VisionEncoderDecoderModel` из библиотеки `transformers` для генерации описаний и `GoogleTranslator`
из `deep-translator` для перевода.

## Пример приложения

![sample1.png](assets%2Fsample1.png)

## Установка

1. **Клонируйте репозиторий:**

   ```bash
   git clone https://github.com/Bagi4-source/imageCaptioning.git
   cd imageCaptioning
   ```

2. **Создайте и активируйте виртуальное окружение:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Для Linux/MacOS
   .venv\Scripts\activate     # Для Windows
   ```

3. **Установите зависимости:**

   Убедитесь, что у вас установлен `pip` и выполните:

   ```bash
   pip install -r requirements.txt
   ```

## Docker

1. **Сборка**

```bash
docker build -t image-captioning -f Dockerfile .
```

2. **Запуск**

docker run -v <путь к папке с картинками>:/app/images image-captioning

```bash
docker run -v images:/app/images image-captioning
```

## Использование

1. **Подготовьте изображения:**

   Поместите изображения, для которых вы хотите сгенерировать подписи, в папку `images`.

2. **Запустите скрипт:**

   Выполните команду:

   ```bash
   python main.py
   ```

   Скрипт обработает все изображения в папке `images`, сгенерирует для них описания и переведёт их на русский язык.

3. **Результаты:**

   Описания изображений будут выведены в консоль.

## Зависимости

- Python 3.8 или выше
- `transformers==4.47.1`
- `torch==2.5.1`
- `torchvision==0.20.1`
- `deep-translator==1.11.4`
- Другие зависимости указаны в `requirements.txt`

## Примечания

- Убедитесь, что у вас есть доступ к интернету для загрузки модели и использования API перевода.
- Если у вас нет модели, загрузите её в папку `model` или укажите путь к модели в коде.

## Лицензия

Этот проект распространяется под лицензией MIT. Подробности смотрите в файле LICENSE.
