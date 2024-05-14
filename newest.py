import os
import json
import cv2
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm
import torch
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, normalize
from PIL import Image
import io
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from transformers import CLIPModel

DEVICE_TO_USE = "cuda" if torch.cuda.is_available() else "cpu"


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


class CLIP64(nn.Module):
    def __init__(self, model_name, pretrained):
        """
        Load a CLIP model and append a PCA layer or a random choice among the head neurons to obtain a 64D vector
        The PCA is obtained on embeddings of plausible labels generated by GPT-3
        """
        super().__init__()
        # Load model and transforms
        with suppress_stdout_stderr():
            model, self.preprocess, _ = create_model_and_transforms(
                model_name,
                pretrained,
                jit=False,
                device=DEVICE_TO_USE
            )
        # Transforms parameters
        self.image_size = self.preprocess.transforms[0].size[0]
        self.mean = self.preprocess.transforms[-1].mean
        self.std = self.preprocess.transforms[-1].std
        # Set encoder
        self.encoder = model.visual.half().eval()

    def forward(self, image):
        """
        The input image is padded and resized to the size required by the CLIP visual encoder
        The PCA layer is then applied to the output of the encode
        """
        # Pad
        h, w = image.size()[2:]
        p_left, p_top = [(max(h, w) - s) // 2 for s in [h, w]]
        p_right, p_bottom = [max(h, w) - (s + pad) for s, pad in zip([h, w], [p_left, p_top])]
        value = 255. * sum(self.mean) / 3
        image = nn.functional.pad(image, [p_top, p_bottom, p_left, p_right], 'constant', value)
        # Resize
        image = resize(image, size=(self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC)
        # Normalize
        image = image.half()
        image = normalize(image, mean=self.mean, std=self.std)
        # Run feature extractor
        features = self.encoder(image.to(DEVICE_TO_USE))[0]
        features = features.unsqueeze(0)
        return features


def get_embedding_from_image(img1, clip_model):
    img1 = np.array(img1)
    img1 = np.expand_dims(img1, axis=0)
    img1 = np.transpose(img1, (0, 3, 1, 2))
    tensor = torch.from_numpy(img1)
    with torch.no_grad():
        emb = clip_model(tensor).cpu().numpy()
    return emb


def create_model_and_transforms(model_name, pretrained, jit, device):
    import open_clip
    model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained,
        jit=jit,
        device=device
    )
    return model, preprocess, None


def extract_features(image_path, clip_model=None):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Извлечение признаков с помощью OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = {
        'area': cv2.contourArea(contours[0]) if contours else 0,
        'x': contours[0][0][0][0] if contours else -1,
        'y': contours[0][0][0][1] if contours else -1,
        'width': cv2.boundingRect(contours[0])[2] if contours else 0,
        'height': cv2.boundingRect(contours[0])[3] if contours else 0,
    }

    # Нормализация координат
    features['x'] /= img.shape[1]
    features['y'] /= img.shape[0]
    features['width'] /= img.shape[1]
    features['height'] /= img.shape[0]

    # Извлечение CLIP embedding
    if clip_model is not None:
        features['clip_embedding'] = get_embedding_from_image(img, clip_model).flatten()

    return features


def process_dataset(dataset_path, annotations_path, clip_model=None):
    data = []

    # Чтение JSON файла с аннотациями
    with open(annotations_path, 'r') as f:
        data_json = json.load(f)

    # Создание словаря для быстрого доступа к аннотациям по image_id
    annotations_by_image_id = {}
    for ann in data_json['annotations']:

        image_id = ann['image_id']
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []
        annotations_by_image_id[image_id].append(ann)

    # Итерация по видео
    for video_data in tqdm(data_json['videos'], desc="Обработка видео"):
        video_id = video_data['id']
        video_name = video_data['name']
        video_width = video_data['width']
        video_height = video_data['height']

        # Путь к папке видео
        video_folder = os.path.join(dataset_path, video_name)

        # Итерация по изображениям в видео
        for image_data in data_json['images']:
            if image_data['video_id'] == video_id:
                image_id = image_data['id']
                filename = image_data['file_name'].split('/')[-1]
                image_path = os.path.join(video_folder, filename)

                # Проверка существования файла изображения
                if os.path.exists(image_path):
                    features = extract_features(image_path, clip_model)

                    # Получение аннотаций для текущего image_id
                    frame_annotations = annotations_by_image_id.get(image_id, [])

                    for obj_annotation in frame_annotations:
                        # Определение target
                        bbox = obj_annotation['bbox']
                        x_min, y_min, w, h = bbox
                        x_max = x_min + w
                        y_max = y_min + h

                        # Проверьте, находится ли точка (features['x'], features['y']) внутри bbox
                        if x_min <= features['x'] <= x_max and y_min <= features['y'] <= y_max:
                            target = 1
                        else:
                            target = 0

                        data.append({
                            'video_id': video_id,
                            'object_id': obj_annotation['track_id'],
                            'image_id': image_id,
                            'video_width': video_width,  # Добавляем ширину видео
                            'video_height': video_height,  # Добавляем высоту видео
                            **features,
                            'target': target
                        })

    df = pd.DataFrame(data)

    # Преобразование CLIP embedding в отдельные признаки
    if clip_model is not None and 'clip_embedding' in df.columns:
        clip_embeddings = pd.DataFrame(df['clip_embedding'].tolist(),
                                       columns=[f'clip_embedding_{i}' for i in range(len(df['clip_embedding'][0]))])
        df = pd.concat([df.drop('clip_embedding', axis=1), clip_embeddings], axis=1)

    return df


# Пути к данным
data_path = "D:/model_data/1-TAO_TRAIN/frames/"
val_path = "D:/model_data/2-TAO_VAL/frames/"
annotations_path = "D:/model_data/1-TAO_TRAIN/annotations/train.json"
validations_path = "D:/model_data/1-TAO_TRAIN/annotations/validation.json"

# Загрузка CLIP модели (если нужно)
clip_model = CLIP64('ViT-H-14', 'laion2b_s32b_b79k')

# Обработка данных
print("Начинаем обработку данных...")
train_df = process_dataset(data_path, annotations_path, clip_model)
print(train_df.head())
val_df = process_dataset(val_path, validations_path, clip_model)
print("Обработка данных завершена.")

# Преобразование данных для CatBoost
print("Создание Pool для CatBoost...")
train_pool = Pool(data=train_df.drop('target', axis=1), label=train_df['target'])
val_pool = Pool(data=val_df.drop('target', axis=1), label=val_df['target'])
print("Pool созданы.")

# Создание и обучение модели CatBoost
print("Обучение модели CatBoost...")
model = CatBoostClassifier(iterations=200, learning_rate=0.01, early_stopping_rounds=20)
model.fit(train_pool, eval_set=val_pool)
print("Обучение модели завершено.")

# Сохранение модели
model_filename = "catboost_tracking_model.cbm"  # Выберите имя файла для модели
model.save_model(model_filename)
print(f"Модель сохранена в файл: {model_filename}")
