import onnxruntime as ort
import cv2
import toml
import numpy as np


configs = toml.load('inferences/configs/config.toml')

detection_session = ort.InferenceSession(configs['detection-model-path'], providers=configs['providers'])
classification_session = ort.InferenceSession(configs['classification-model-path'], providers=configs['providers'])


def normalize(inputs):
    return inputs / 255.0


def letterbox(image, size, padding):
    image_w = image.shape[1]
    image_h = image.shape[0]

    longest_side = max(image_w, image_h)

    x1 = (longest_side - image_w) >> 1
    y1 = (longest_side - image_h) >> 1

    x2 = x1 + image_w
    y2 = y1 + image_h

    background = np.full((longest_side, longest_side, 3), padding, dtype=np.uint8)
    background[y1:y2, x1:x2] = image

    return cv2.resize(background, (size, size), interpolation=cv2.INTER_LINEAR)


def preprocess(image):
    inputs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))

    if configs['precision'] == 'fp16':
        inputs = normalize(inputs).astype(np.float16)
    else:
        inputs = normalize(inputs).astype(np.float32)

    return np.expand_dims(inputs, axis=0)


def get_valid_outputs(outputs):
    valid_outputs = outputs[outputs[:, 4] > configs['conf-threshold']]

    bboxes = valid_outputs[:, 0:4]
    scores = valid_outputs[:, 4]

    return bboxes.astype(np.int32), scores.astype(np.float32)


def non_max_suppression(outputs):
    bboxes, scores = get_valid_outputs(outputs)

    bboxes[:, 0] -= bboxes[:, 2] >> 1
    bboxes[:, 1] -= bboxes[:, 3] >> 1

    for index in cv2.dnn.NMSBoxes(bboxes, scores, configs['conf-threshold'], configs['iou-threshold'], eta=0.5):
        x1 = bboxes[index, 0]
        y1 = bboxes[index, 1]

        x2 = bboxes[index, 2] + x1
        y2 = bboxes[index, 3] + y1

        yield x1, y1, x2, y2


def detection_inference(image):
    detections = detection_session.run(['output0'], {'images': preprocess(image)})
    detections = detections[0]
    detections = detections.squeeze().transpose()

    return non_max_suppression(detections)


def classification_inference(image):
    classes_outputs = classification_session.run(['output0'], {'images': preprocess(image)})
    classes_outputs = classes_outputs[0]
    classes_outputs = classes_outputs.squeeze()

    return np.argmax(classes_outputs)


def classify(image):
    return classification_inference(letterbox(image, size=64, padding=0)).item()


def inference(image):
    letterbox_image = letterbox(image, size=640, padding=255)

    for bbox in detection_inference(letterbox_image):
        x1 = bbox[0].item()
        y1 = bbox[1].item()
        x2 = bbox[2].item()
        y2 = bbox[3].item()

        yield {'bbox': [x1, y1, x2, y2], 'classes': classify(letterbox_image[y1:y2, x1:x2])}
