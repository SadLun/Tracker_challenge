# Tracker_challenge
Этот репозиторий содержит файлы с экспериментными трекерами

**validate_old** -- эксперимент с KCF трекером из библиотеки OpenCV; 
**newest** -- модель на основе градиентного бустинга (CatBoost), обученная на датасете TAO Challenge; 
**main** -- эскперимент с использованием предыдущих наработок и добавлением обученной модели

Код запускать из под Linux или Windows (работает из под WSL2).

**Скачать в папку валидационые файлы (всего 4 штуки):**
vot initialize tests/multiobject --workspace /mnt/i/2023_03_VOT_Challenge/input/romanWorkspaceSmall

**Скачать в папку тестовые файлы (всего 144 видео)**
vot initialize vots2023 --workspace /mnt/i/2023_03_VOT_Challenge/input/romanWorkspace

В созданном workspace лежит файл trackers.ini

Его примерное содержимое:

[SRZLT_HSE_IPPM_ClipSegmentAnything]  # <tracker-name>
label = PyNCC
protocol = traxpython
command = ZFTurbo_HSE_IPPM_tracker_SegmentAnything_and_CLIP
paths = /mnt/i/2023_03_VOT_Challenge/vot-challenge-2023/tracker_v1
;env_PATH = <additional-env-paths>;${PATH}

В нем нужно поменять **command** - имя питон файла с трекером. И прописать путь до этого питон файла в переменной **paths**.

**Далее когда все настроено можно запустить прогон видео через трекер и получить результаты для валидации:**

vot evaluate --workspace /mnt/i/2023_03_VOT_Challenge/input/romanWorkspaceSmall SRZLT_HSE_IPPM_ClipSegmentAnything

**Когда прогон закончится то можно запустить анализ который посчитает метрики качества:**

vot analysis --workspace /mnt/i/2023_03_VOT_Challenge/input/romanWorkspaceSmall SRZLT_HSE_IPPM_ClipSegmentAnything

**Запуск основного прогона** (144 видео - это может быть медленно). И можно выбрать на каком GPU считать если их несколько (в примере расчет будет на GPU номер 2):

CUDA_VISIBLE_DEVICES=2 vot evaluate --workspace /mnt/i/2023_03_VOT_Challenge/input/romanWorkspace SRZLT_HSE_IPPM_ClipSegmentAnything

**Для упаковки результата для отправки на лидерборд:**

vot pack --workspace /mnt/i/2023_03_VOT_Challenge/input/romanWorkspace SRZLT_HSE_IPPM_ClipSegmentAnything

