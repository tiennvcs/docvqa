import json
import os
from tqdm import tqdm
import numpy as np
import cv2


def draw_ocr(base_dir, img_path, draw_text=False):
    img = cv2.imread(os.path.join(base_dir, 'documents', img_path))
    line_img = img.copy()
    word_img = img.copy()
    ocr_path = os.path.join(base_dir, 'ocr_results', img_path.split(".")[0]+'.json')
    with open(ocr_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    recognitionResults = data['recognitionResults'][0]
    lines = recognitionResults['lines']
    for line in lines:
        bbox = line['boundingBox']
        text = line['text']
        vertices = np.array([[bbox[i], bbox[i+1]] for i in range(0, len(bbox)-1, 2)], np.int32).reshape((-1, 1, 2))
        line_img = cv2.polylines(line_img, [vertices], True, (0, 255, 0), 2)
        if draw_text:
            line_img = cv2.putText(line_img, text, vertices[0][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for word_ocr in line['words']:
            word_bbox = word_ocr['boundingBox']
            word_text = word_ocr['text']
            word_vertices = np.array([[word_bbox[i], word_bbox[i+1]] for i in range(0, len(word_bbox)-1, 2)], np.int32).reshape((-1, 1, 2))
            word_img = cv2.polylines(word_img, [word_vertices], True, (0, 255, 255), 2)
            if draw_text:
                word_img = cv2.putText(word_img, word_text, word_vertices[0][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return line_img, word_img


def draw_entire(base_dir):
    img_paths = os.listdir(os.path.join(base_dir, 'documents'))
    ocr_paths = os.listdir(os.path.join(base_dir, 'ocr_results'))
    output_dir = os.path.join(base_dir, 'ocr_documents/')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    line_output_dir = os.path.join(output_dir, 'line')
    if not os.path.exists(line_output_dir):
        os.mkdir(line_output_dir)
    word_output_dir = os.path.join(output_dir, 'word')
    if not os.path.exists(word_output_dir):
        os.mkdir(word_output_dir)

    for img_path in tqdm(img_paths):
        if not img_path in ocr_paths:
            pass
        line_img, word_img = draw_ocr(img_path=img_path, base_dir=base_dir)
        cv2.imwrite(os.path.join(line_output_dir, img_path), line_img)
        cv2.imwrite(os.path.join(word_output_dir, img_path), word_img)


if __name__ == '__main__':
    base_dir = 'visualize_docvqa/static/val/'
    draw_entire(base_dir=base_dir)
