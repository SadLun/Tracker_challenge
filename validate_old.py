import sys
import vot
import cv2
import numpy as np

class MultiObjectTracker:
    def __init__(self):
        self.trackers = []
        self.lost_trackers = []

    def initialize(self, image, masks):
        self.trackers = []
        self.lost_trackers = []
        for mask in masks:
            x, y, w, h = cv2.boundingRect(mask)
            tracker = cv2.legacy.TrackerKCF.create()
            success = tracker.init(image, (x, y, w, h))
            if success:
                self.trackers.append(tracker)
                self.lost_trackers.append(False)
            else:
                self.trackers.append(None)
                self.lost_trackers.append(True)

    def update(self, image):
        bboxes = []
        for i, tracker in enumerate(self.trackers):
            if tracker is not None:
                success, bbox = tracker.update(image)
                if success and self._is_bbox_valid(image, bbox):
                    bboxes.append(bbox)
                    self.lost_trackers[i] = False
                else:
                    bboxes.append(None)
                    self.lost_trackers[i] = True
            else:
                bboxes.append(None)
                self.lost_trackers[i] = True
        return bboxes

    def _is_bbox_valid(self, image, bbox):
        x, y, w, h = bbox
        return 0 <= x < image.shape[1] and 0 <= y < image.shape[0] and w > 0 and h > 0 and (x + w) <= image.shape[1] and (y + h) <= image.shape[0]


handle = vot.VOT("mask", multiobject=True)
objects = handle.objects()
print('Amount of obj: {}'.format(len(objects)))
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)


image = cv2.imread(imagefile)
tracker = MultiObjectTracker()
tracker.initialize(image, objects)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)

    bboxes = tracker.update(image)
    results = []
    for bbox, mask in zip(bboxes, objects):
        if bbox is not None:
            x, y, w, h = bbox
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[int(y):int(y+h), int(x):int(x+w)] = 1
            results.append(mask)
        else:
            if not np.any(mask):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            results.append(mask)
    handle.report(results)
