import torch
import os
import json
import subprocess
import logging
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import LayoutLMv2FeatureExtractor
from config import MODEL_CHECKPOINT, features, DEBUG, BATCH_SIZE
import cv2
import numpy as np
feature_extractor = LayoutLMv2FeatureExtractor()
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


def subfinder(words_list, answer_list):
    matches = []
    start_indices = []
    end_indices = []
    for idx, i in enumerate(range(len(words_list))):

        if words_list[i] == answer_list[0] and words_list[i:i + len(answer_list)] == answer_list:
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    if len(matches) != 0:
        return matches[0], start_indices[0], end_indices[0]
    else:
        return None, 0, 0


def get_ocr_words_and_boxes(examples):

    # get a batch of document images
    images = [cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB) for image_file in examples['image']]

    # resize every image to 224x224 + apply tesseract to get words + normalized boxes
    encoded_inputs = feature_extractor(images)


    examples['image'] = encoded_inputs.pixel_values
    ####################################################################################################
    # DEBUG
    # 3x224x224 -> 224x224x3 : 
    # print(examples['image'][0].transpose(1, 2, 0).shape)
    # print("SAVING RESIZED IMAGE TO CHECK")
    # cv2.imwrite('./test_code/image_before_encoding.png', images[0])
    # cv2.imwrite('./test_code/image_after_encoding.png', examples['image'][0].transpose(1, 2, 0))
    # exit(0)
    # DONE DEBUG 
    ####################################################################################################

    examples['words'] = encoded_inputs.words
    examples['boxes'] = encoded_inputs.boxes

    return examples


def get_avail_ocr_feature(examples):
    
    # get a batch of document images
    images = [cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB) for image_file in examples['image']]

    # resize every image to 224x224 + apply tesseract to get words + normalized boxes
    images = [cv2.resize(src=img, dsize=(224, 224), interpolation=cv2.INTER_AREA) for img in images]

    # Reshape image to disized shape
    images = [img.transpose(2, 0, 1) for img in images]
    
    # Load ocr info
    examples['image'] = images
    examples['words'] = []
    examples['boxes'] = []
    # Loop through each image
    for i in range(len(examples['image'])):
        words_img = []
        boxes_img = []
        with open(examples['ocr_output_file'][i], 'r') as f:
            data = json.load(f)   # data = {"status": [], "recognitionResults": []}
            recognitionResults = data['recognitionResults']
            # Loop through each recognition line
            for reg_result in recognitionResults:
                lines = reg_result['lines']
                for line in lines:
                    for word_info in line['words']:
                        word_info['boundingBox'] = (word_info['boundingBox'])
                        x_min = np.min(word_info['boundingBox'][0:-1:2])
                        y_min = np.min(word_info['boundingBox'][1:-1:2])
                        x_max = np.max(word_info['boundingBox'][0:-1:2])
                        y_max = np.max(word_info['boundingBox'][1:-1:2])
                        words_img.append(word_info['text'])
                        boxes_img.append(normalize_bbox(bbox=[x_min, y_min, x_max, y_max], 
                            width=reg_result['width'], height=reg_result['height']))
            examples['words'].append(words_img)
            examples['boxes'].append(boxes_img)

    return examples


def encode_dataset(examples, max_length=512):

    questions = examples['question']
    words = examples['words']
    boxes = examples['boxes']
    
    encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)

    # next, add start_positions and end_positions
    start_positions = [0]*BATCH_SIZE
    end_positions = [0]*BATCH_SIZE
    answers = examples['answers']
    # for every example in the batch:
    for batch_index in range(len(answers)):
        cls_index = encoding.input_ids[batch_index].index(tokenizer.cls_token_id)
        # try to find one of the answers in the context, return first match
        words_example = [word.lower() for word in words[batch_index]]
        for answer in answers[batch_index]:
            match, word_idx_start, word_idx_end = subfinder(words_example, answer.lower().split())
            if match != None:
                break
    
        if match:
            sequence_ids = encoding.sequence_ids(batch_index)
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(encoding.input_ids[batch_index]) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            
            word_ids = encoding.word_ids(batch_index)[token_start_index:token_end_index+1]
            for id in word_ids:
                if id == word_idx_start:
                    start_positions[batch_index] = token_start_index
                    break
                else:
                    token_start_index += 1

            for id in word_ids[::-1]:
                if id == word_idx_end:
                    end_positions[batch_index] = token_end_index
                    break
                else:
                    token_end_index -= 1
        else:
            start_positions[batch_index] = cls_index
            end_positions[batch_index]   = cls_index

    encoding['image'] = examples['image']
    encoding['start_positions'] = start_positions
    encoding['end_positions'] = end_positions

    return encoding


def gather_ocr_file(ocr_dir, img_paths):
    
    imgs      = [os.path.basename(img) for img in img_paths]
    # Gether available ocr file inside ocr_dir
    ocr_files = [os.path.join(ocr_dir, img.split(".")[0]+'.json') for img in imgs]
    return ocr_files
    

def load_and_process_data(data_dir, batch_size, num_workers):
    label_file = None
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            label_file = os.path.join(data_dir, file)
            break
    with open(label_file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data['data'])
    # df = df.sort_values('image')
    df['image'] = [os.path.join(data_dir, img_file) for img_file in df['image']]

    # Check availabel OCR file
    ocr_dir = [subdir for subdir in os.listdir(data_dir) if "ocr" in subdir]
    if len(ocr_dir) > 0:
        if 'our_output_file' in df.columns:
            df['ocr_output_file'] = [os.path.join(data_dir, ocr_file) for ocr_file in df['ocr_output_file']]
        else:
            full_ocr_dir = os.path.join(data_dir, ocr_dir[0])
            df['ocr_output_file'] = gather_ocr_file(full_ocr_dir, df['image'])
        dataset = Dataset.from_pandas(df.iloc[:DEBUG])
        dataset_with_ocr = dataset.map(get_avail_ocr_feature, batched=True, batch_size=batch_size)
    else:
        # Extract feature by OCR
        dataset = Dataset.from_pandas(df.iloc[:DEBUG])
        dataset_with_ocr = dataset.map(get_ocr_words_and_boxes, batched=True, batch_size=batch_size)
    encoded_dataset = dataset_with_ocr.map(encode_dataset, batched=True, batch_size=batch_size,
					remove_columns=dataset_with_ocr.column_names, features=features)
    encoded_dataset.set_format(type="torch")
    dataloader = torch.utils.data.DataLoader(encoded_dataset, shuffle=True,
                                            batch_size=batch_size, num_workers=num_workers)
    return dataloader



def load_feature_from_file(path, batch_size=2, num_workers=4):
    encoded_dataset = torch.load(path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else "cpu"))
    encoded_dataset.set_format(type="torch")
    dataloader = torch.utils.data.DataLoader(encoded_dataset, shuffle=True, pin_memory=True,
                                            batch_size=batch_size, num_workers=num_workers)
    return dataloader



def normalize_bbox(bbox, width, height):
     return [
         int(1000 * (bbox[0] / width)),
         int(1000 * (bbox[1] / height)),
         int(1000 * (bbox[2] / width)),
         int(1000 * (bbox[3] / height)),
    ]


def create_logger(file_path):
    log = logging.getLogger(file_path)
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(file_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    if not 'loss' in file_path:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)

    return log


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return np.array(gpu_memory)
