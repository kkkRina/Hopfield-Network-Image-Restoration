from HopfildNN import HopfildNN
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def convert_to_bmp(source_path, destination_path):
    with Image.open(source_path) as img:
        img = img.resize((64, 64))
        img = img.convert("L")
        img = img.point(lambda p: 255 if p > 127 else 0)
        img.save(destination_path, "BMP")

def add_noise_to_image(image_array, noise_ratio=0.1):

    noisy_image = image_array.copy()
    total_pixels = noisy_image.size
    num_noisy_pixels = int(total_pixels * noise_ratio)

    indices = np.random.choice(total_pixels, num_noisy_pixels, replace=False)
    for index in indices:
        noisy_image.flat[index] = 255 - noisy_image.flat[index]  # инвертируем пиксель

    return noisy_image


def split_and_convert_data(input_folder, output_folder):
    train_folder = os.path.join(output_folder, "train")
    test_folder = os.path.join(output_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        train_class_folder = os.path.join(train_folder, class_name)
        test_class_folder = os.path.join(test_folder, class_name)
        os.makedirs(train_class_folder, exist_ok=True)
        os.makedirs(test_class_folder, exist_ok=True)

        files = [f for f in os.listdir(class_path) if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))]

        for file in files:
            source_path = os.path.join(class_path, file)
            train_destination_path = os.path.join(train_class_folder, os.path.splitext(file)[0] + ".bmp")
            test_destination_path = os.path.join(test_class_folder, os.path.splitext(file)[0] + ".bmp")

            # Создаем тренировочное изображение (без шума)
            convert_to_bmp(source_path, train_destination_path)

            # Создаем тестовое изображение (с шумом)
            with Image.open(source_path) as img:
                img = img.resize((64, 64))
                img = img.convert("L")
                img = img.point(lambda p: 255 if p > 127 else 0)
                img_array = np.asarray(img)
                noisy_image_array = add_noise_to_image(img_array)
                noisy_image = Image.fromarray(noisy_image_array)
                noisy_image.save(test_destination_path, "BMP")

    print("данные преобразовани...")


def load_images_from_folder(folder):
    #Загружает изображения из папки, конвертирует в черно-белые и преобразует в бинарные векторы.

    data = []

    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            if file.lower().endswith(".bmp"):
                file_path = os.path.join(class_path, file)
                with Image.open(file_path) as img:
                    img = img.convert("L")
                    img = img.point(lambda p: 1 if p > 127 else -1)
                    img_array = np.asarray(img)
                    data.append(img_array.flatten())  # Преобразуем в одномерный вектор
    return np.array(data)

if __name__ == "__main__":

    input_folder = "extracted_images"
    output_folder = "bmp_images"

    split_and_convert_data(input_folder, output_folder)

    train_folder = os.path.join(output_folder, "train")
    test_folder = os.path.join(output_folder, "test")

    train_data = load_images_from_folder(train_folder)
    test_data = load_images_from_folder(test_folder)

    print("обучени...")
    h = HopfildNN()
    h.train(train_data)


    def visualize_predictions(predictions, train_data, title="восстановленни"):

        fig, axes = plt.subplots(1, len(predictions), figsize=(15, 5))

        if len(predictions) == 1:
            axes = [axes]

        for i, idx in enumerate(predictions):
            image = train_data[idx].reshape(64, 64)
            axes[i].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[i].axis("off")
            axes[i].set_title(f"{i + 1}")

        plt.suptitle(title)
        plt.show()

    print("тестировани...")
    predictions = h.predict(test_data)

    visualize_predictions(predictions[:5], train_data)





