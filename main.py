#!/usr/bin/python
# coding: utf-8
__author__ = 'ZFTurbo: https://github.com/ZFTurbo'

# Note: because initialization is slow and downloading of weights also slow, please change line in module vot.tracker.trax.py
# in function `trax_python_adapter` line
# return TraxTrackerRuntime(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars, arguments=arguments, socket=socket, restart=restart)
# to
# return TraxTrackerRuntime(tracker, command, log=log, timeout=30000, linkpaths=linkpaths, envvars=envvars, arguments=arguments, socket=socket, restart=restart)
# to find location of vot.tracker.trax.py use "python -m site"

# Link 1: https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/playground/ImageBind_SAM
# Link 2: https://github.com/facebookresearch/segment-anything
# Link 3: https://github.com/mlfoundations/open_clip

if __name__ == '__main__':
    import os

    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


# import vot
# import numpy as np
# import os
# from contextlib import contextmanager, redirect_stderr, redirect_stdout
# from os import devnull
import gzip
# from skimage.measure import regionprops
# import contextlib
# from segment_anything import build_sam, SamAutomaticMaskGenerator
import pickle
# import cv2
# import requests
# import torch
# from torch import nn
# from sklearn.decomposition import PCA
# from torchvision.transforms import InterpolationMode
# from torchvision.transforms.functional import resize, normalize
# from open_clip import create_model_and_transforms, tokenize
# from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, normalize, to_tensor
from open_clip import create_model_and_transforms, tokenize
from sklearn.metrics.pairwise import cosine_similarity
from skimage.measure import regionprops
from segment_anything import build_sam, SamAutomaticMaskGenerator, SamPredictor
import vot
from catboost import CatBoostClassifier, Pool
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from tqdm import tqdm


LIMIT_PROCESSED_FRAMES = 500000
DEVICE_TO_USE = 'cuda' if torch.cuda.is_available() else 'cpu'

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


class CLIP64(nn.Module):
    def __init__(self, model_name, pretrained):
        super().__init__()
        with suppress_stdout_stderr():
            model, self.preprocess, _ = create_model_and_transforms(
                model_name,
                pretrained,
                jit=False,
                device=DEVICE_TO_USE
            )
        self.image_size = self.preprocess.transforms[0].size[0]
        self.mean = self.preprocess.transforms[-1].mean
        self.std = self.preprocess.transforms[-1].std
        self.encoder = model.visual.half().eval()

    def forward(self, image):
        # Pad
        h, w = image.shape[1:] # Исправлено: h, w = image.size()[2:] -> h, w = image.shape[1:]
        p_left, p_top = [(max(h, w) - s) // 2 for s in [h, w]]
        p_right, p_bottom = [max(h, w) - (s + pad) for s, pad in zip([h, w], [p_left, p_top])]
        value = 255. * sum(self.mean) / 3
        image = nn.functional.pad(image, [p_left, p_right, p_top, p_bottom], 'constant', value)
        # Resize
        image = resize(image, size=(self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC)
        # Normalize
        image = image.to(torch.float) / 255. # Исправлено: image = image.half() / 255. -> image = image.to(torch.float) / 255.
        image = normalize(image, mean=self.mean, std=self.std)
        # Run feature extractor
        features = self.encoder(image.unsqueeze(0).to(DEVICE_TO_USE))[0]
        features = features.unsqueeze(0)
        return features


def get_embedding_from_image(img1, clip_model):
    img1 = np.array(img1)
    tensor = to_tensor(img1).to(DEVICE_TO_USE) # Исправлено: tensor = torch.from_numpy(img1).to(DEVICE_TO_USE)
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


def extract_features(image, clip_model=None): # Исправлено: def extract_features(image_path, clip_model=None):
    # Извлечение признаков с помощью OpenCV
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Исправлено: cv2.COLOR_BGR2GRAY -> cv2.COLOR_RGB2GRAY
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
    features['x'] /= image.shape[1] # Исправлено: img.shape[1] -> image.shape[1]
    features['y'] /= image.shape[0] # Исправлено: img.shape[0] -> image.shape[0]
    features['width'] /= image.shape[1] # Исправлено: img.shape[1] -> image.shape[1]
    features['height'] /= image.shape[0] # Исправлено: img.shape[0] -> image.shape[0]

    # Извлечение CLIP embedding
    if clip_model is not None:
        features['clip_embedding'] = get_embedding_from_image(image, clip_model).flatten()

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
        for i in range(len(df['clip_embedding'][0])):
            df[f'clip_embedding_{i}'] = df['clip_embedding'].apply(lambda x: x[i])
        df = df.drop('clip_embedding', axis=1)

    return df


def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3), protocol=4)


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'), protocol=4)


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


# class CLIP64(nn.Module):
#     def __init__(self, model_name, pretrained):
#         """
#         Load a CLIP model and append a PCA layer or a random choice among the head neurons to obtain a 64D vector
#         The PCA is obtained on embeddings of plausible labels generated by GPT-3
#         """
#         super().__init__()
#
#         # Load model and transforms
#         with suppress_stdout_stderr():
#             model, transforms, _ = create_model_and_transforms(
#                 model_name,
#                 pretrained,
#                 jit=False,
#                 device=DEVICE_TO_USE
#             )
#
#         # Transforms parameters
#         self.image_size = transforms.transforms[0].size[0]
#         self.mean = transforms.transforms[-1].mean
#         self.std = transforms.transforms[-1].std
#
#         # Set encoder
#         self.encoder = model.visual.half().eval()
#
#     def forward(self, image):
#         """
#         The input image is padded and resized to the size required by the CLIP visual encoder
#         The PCA layer is then applied to the output of the encode
#         """
#
#         # Pad
#         h, w = image.size()[2:]
#         p_left, p_top = [(max(h, w) - s) // 2 for s in [h, w]]
#         p_right, p_bottom = [max(h, w) - (s + pad) for s, pad in zip([h, w], [p_left, p_top])]
#         value = 255. * sum(self.mean) / 3
#         image = nn.functional.pad(image, [p_top, p_bottom, p_left, p_right], 'constant', value)
#
#         # Resize
#         image = resize(image, size=(self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC)
#
#         # Normalize
#         image = image.half()
#         image /= 255.
#         image = normalize(image, mean=self.mean, std=self.std)
#
#         # Run feature extractor
#         features = self.encoder(image.to(DEVICE_TO_USE))[0]
#         features = features.unsqueeze(0)
#
#         return features


# def get_embedding_from_image(img1, clip_model):
#     img1 = np.array(img1)
#     img1 = np.expand_dims(img1, axis=0)
#     img1 = np.transpose(img1, (0, 3, 1, 2))
#     tensor = torch.from_numpy(img1)
#     with torch.no_grad():
#         emb = clip_model(tensor).cpu().numpy()
#     return emb


def save_debug_info(txt):
    root_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    out = open(root_path + "debug.txt", "a")
    out.write(txt)
    out.close()


def save_debug_image(img, suffix):
    root_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    cv2.imwrite(root_path + 'debug_{}.jpg'.format(suffix), img)


mask_generator = None
clip_model = None


class ZFTracker(object):
    def __init__(self, image, masks_list):
        global mask_generator
        global clip_model
        tracker_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
        device = DEVICE_TO_USE if torch.cuda.is_available() else "cpu"

        # Segment Anything
        if mask_generator is None:
            weights_path = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            local_path = tracker_dir + "sam_vit_h_4b8939.pth"
            if not os.path.isfile(local_path):
                torch.hub.download_url_to_file(weights_path, local_path, progress=False)
            sam = build_sam(checkpoint=local_path).to(device)
            mask_generator = SamPredictor(sam)
        # Open CLIP
        if clip_model is None:
            clip_model = CLIP64('ViT-H-14', 'laion2b_s32b_b79k')
            clip_model.eval()
            clip_model.to(device)

        # Загрузка модели CatBoost
        self.catboost_model = CatBoostClassifier()
        self.catboost_model.load_model("catboost_tracking_model.cbm")

        self.image = image.copy()
        self.emb_template = []
        self.template_masked = []
        for object_id, mask in enumerate(masks_list):
            mask_full = np.zeros(image.shape[:2], dtype=np.uint8)
            mask_full[:mask.shape[0], :mask.shape[1]] = mask
            mask = mask_full
            bboxes = regionprops(255 * mask)
            if len(bboxes) > 1:
                print('Many masks found!')
                exit()
            prop = bboxes[0]
            # Left only mask
            image_masked = image.copy()
            image_masked[mask == 0] = 0
            self.mask = mask.copy()
            self.template_masked.append(
                image_masked[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]].copy()
            )
            self.emb_template.append(
                get_embedding_from_image(self.template_masked[-1], clip_model)
            )
            self.current_image = 0

    def track(self, image):
        if self.current_image < LIMIT_PROCESSED_FRAMES:
            masks = mask_generator.generate(image)
            # Используйте маску из предыдущего кадра в качестве input_prompts
            input_boxes = torch.tensor([convert_box_xywh_to_xyxy(mask["bbox"]) for mask in masks], dtype=torch.float).to(DEVICE_TO_USE)
            transformed_boxes = mask_generator.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            masks, _, _ = mask_generator.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks = masks.cpu().numpy() # Добавлено: masks = masks.cpu().numpy()
            embeddings = []
            features_for_prediction = []
            for mask in masks:
                image_masked = image.copy()
                image_masked[mask.astype(np.uint8) == 0] = 0
                x1, y1, x2, y2 = convert_box_xywh_to_xyxy(mask["bbox"])
                try:
                    img1 = image_masked[y1:y2, x1:x2]
                except Exception as e:
                    save_debug_info("{} {} {} {} {} {}\n".format(self.current_image, x1, y1, x2, y2, mask["bbox"]))
                    img1 = np.zeros((10, 10, 3), dtype=np.uint8)
                features = extract_features(img1, clip_model)
                features_for_prediction.append({
                    'video_width': image.shape[1],
                    'video_height': image.shape[0],
                    **features
                })
                emb = get_embedding_from_image(img1, clip_model)
                embeddings.append(emb)

            prediction_df = pd.DataFrame(features_for_prediction)

            # Преобразование CLIP embedding в отдельные признаки
            if clip_model is not None:
                for i in range(len(prediction_df['clip_embedding'][0])):
                    prediction_df[f'clip_embedding_{i}'] = prediction_df['clip_embedding'].apply(lambda x: x[i])
                prediction_df = prediction_df.drop('clip_embedding', axis=1)

            # Предсказание CatBoost
            predictions = self.catboost_model.predict_proba(prediction_df)[:, 1]
            print(predictions)

            X = np.concatenate(embeddings, axis=0)
            Y = np.concatenate(self.emb_template, axis=0)
            distances = cosine_similarity(X, Y)
            best_indexes = np.argmax(distances, axis=0)
            mask_list = []
            for object_id, best_index in enumerate(best_indexes):
                print('Object: {} Masks found: {} Best index: {} Max value: {:.6f}'.format(
                    object_id,
                    len(masks),
                    best_index,
                    distances[best_index, object_id])
                )
                mask_chosen = masks[best_index].astype(np.uint8).copy() # Исправлено: mask_chosen = masks[best_index]["segmentation"].astype(np.uint8).copy()
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[...] = mask_chosen
                mask_list.append(mask)
        else:
            mask_list = []
            for i in range(len(self.emb_template)):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask_list.append(mask)
        self.current_image += 1
        return mask_list


handle = vot.VOT("mask", multiobject=True)
objects = handle.objects()
imagefile = handle.frame()
image = cv2.imread(imagefile)
print('Image path: {} Objects: {}'.format(imagefile, len(objects)))
tracker = ZFTracker(image, objects)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    mask_list = tracker.track(image)
    handle.report(mask_list)
