import torch
import os
import json
import subprocess
import logging
import cv2
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import LayoutLMv2Processor, LayoutLMv2FeatureExtractor
from config import BASE_MODEL_CHECKPOINT, LARGE_MODEL_CHECKPOINT, features, DEBUG, BATCH_SIZE, CHECK
from datasets import Dataset
import torch.distributed as dist
from PIL import Image


processor   = LayoutLMv2Processor.from_pretrained(BASE_MODEL_CHECKPOINT, revision="no_ocr")
feature_extractor = LayoutLMv2FeatureExtractor()


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
    examples['words'] = encoded_inputs.words
    examples['boxes'] = encoded_inputs.boxes

    return examples


def normalize_bbox(bbox, width, height):
     return [
         int(1000 * (bbox[0] / width)),
         int(1000 * (bbox[1] / height)),
         int(1000 * (bbox[2] / width)),
         int(1000 * (bbox[3] / height)),
    ]


def read_ocr_annotation(file_path, shape):
    words_img = []
    boxes_img = []
    width, height = shape
    with open(file_path, 'r') as f:
        data = json.load(f)   # data = {"status": [], "recognitionResults": []}
        try:
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
        except:
            if not 'WORD' in data.keys():
                print("! Ignore ", file_path)
                return [], []
                
            for word in data['WORD']:
                text = word['Text']
                bbox = word['Geometry']['BoundingBox']
                bbox = [bbox['Left']*width, bbox['Top']*height, 
                        (bbox['Left'] + bbox['Width'])*width, 
                        (bbox['Top'] + bbox['Height'])*height]
                nl_bbox = normalize_bbox(bbox=bbox, width=width, height=height)
                words_img.append(text)
                boxes_img.append(nl_bbox)
    
    return (words_img, boxes_img)


def get_avail_ocr_feature(examples):
    
    # get a batch of document images
    images  = [cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB) for image_file in examples['image']]
    org_shapes  = [img.shape[0:2] for img in images]

    # resize every image to 224x224 + apply tesseract to get words + normalized boxes
    images = [cv2.resize(src=img, dsize=(224, 224), interpolation=cv2.INTER_AREA) for img in images]

    # Reshape image to disized shape
    images = [img.transpose(2, 0, 1) for img in images]
    
    # Load ocr info
    examples['image'] = images
    examples['words'] = []
    examples['bbox'] = []
    # Loop through each image
    for i in range(len(examples['image'])):
        words_img, boxes_img = read_ocr_annotation(file_path=examples['ocr_output_file'][i], shape=org_shapes[i])
        examples['words'].append(words_img)
        examples['bbox'].append(boxes_img)

    return examples


def encode_dataset(examples, max_length=512):

    images         = [Image.open(image_file).convert("RGB") for image_file in examples['image']]
    org_shapes     = [img.size[0:2] for img in images]

    words          = []
    bbox           = []
    for i in range(len(images)):
        words_img, boxes_img = read_ocr_annotation(file_path=examples['ocr_output_file'][i], shape=org_shapes[i])
        words.append(words_img)
        bbox.append(boxes_img)

    questions  = examples['question']
    encoding   = processor(images, questions, words, bbox, max_length=max_length, padding="max_length", truncation=True)

    # next, add start_positions and end_positions
    start_positions = [0]*BATCH_SIZE
    end_positions   = [0]*BATCH_SIZE

    answers = examples['answers']
    
    # for every example in the batch:
    for idx in range(len(answers)):
        cls_index = encoding.input_ids[idx].index(processor.tokenizer.cls_token_id)

        words_example = [word.lower() for word in words[idx]]

        for answer in answers[idx]:
            match, word_idx_start, word_idx_end = subfinder(words_example, answer.lower().split())
            if match != None:
                break
    
        if match != None:
            sequence_ids = encoding.sequence_ids(idx)
            
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(encoding.input_ids[idx]) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            word_ids = encoding.word_ids(idx)[token_start_index:token_end_index+1]
            for id in word_ids:
                if id == word_idx_start:
                    start_positions[idx] = token_start_index
                    break
                else:
                    token_start_index += 1

            for id in word_ids[::-1]:
                if id == word_idx_end:
                    end_positions[idx] = token_end_index
                    break
                else:
                    token_end_index -= 1
        else:
            start_positions[idx] = cls_index
            end_positions[idx] = cls_index


    encoding['start_positions'] = start_positions
    encoding['end_positions']   = end_positions
    encoding['question_id']     = examples['questionId']

    # Store to debug whether feature extraction is correct or not :) 
    if CHECK:
        for idx in range(len(answers)):
            dict_info = {   
                'question_id'          : examples['questionId'][idx],
                'image'                : os.path.basename(examples['image'][idx]),
                'question'             : questions[idx],
                'answer'               : answers[idx],
                'construct_answer'     : processor.tokenizer.decode(encoding.input_ids[idx][start_positions[idx]:end_positions[idx]+1]),  
                'words'                : words[idx],
            }
            with open(os.path.join('./runs/debug/', 'extract_features', 'docvqa', str(dict_info['question_id'])+'.json'), 'w', encoding='utf-8') as f:
                json.dump(dict_info, f, indent=4)

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
    dataset = Dataset.load_from_disk(path)
    dataset.set_format(type="torch")
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, pin_memory=True,
                                            batch_size=batch_size, num_workers=num_workers)
    return dataloader


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
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return np.array(gpu_memory)


def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend, rank=rank, world_size=size)


def find_highest_score_answer(start_scores, end_scores):
    start_indices = []
    end_indices   = []
    for start_idx, end_idx in zip(start_scores, end_scores):
        highest_start  = np.argmax(start_idx)
        highest_end    = np.argmax(end_idx)
        # Find the subarray that pick the highest total score for which end >= start.
        if highest_start < highest_end:
            highest_end = highest_start + 1 
        start_indices.append(highest_start)
        end_indices.append(highest_end)

    return (start_indices, end_indices)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def examinize_sample(encoding, processor, words, boxes):
    print("Keys: {}".format(encoding.keys()))
    # for key, value in encoding.items():
    #     print("{} : {}".format(key, value))
    
    input_ids = encoding['input_ids']
    token_type_ids = encoding['token_type_ids']
    print("Length of input_ids: {}".format(len(input_ids[0])))
    print(processor.tokenizer.decode(input_ids[0]))
    print("Length of token_type_ids: {}".format(len(token_type_ids[0])))
    print(token_type_ids[0])
    attention_mask = encoding['attention_mask']
    print("Length of attention_mask: {}".format(len(attention_mask[0])))
    print(attention_mask[0])
    atter_bbox = encoding['bbox']
    print("Length of after bbox: {}".format(len(atter_bbox[0])))
    print(atter_bbox[0])
    print("Length of before bbox: {}".format(len(boxes[0])))
    print(boxes[0])
    print("Length of before text: {}".format(len(words[0])))
    print(words[0])
    print("Reshape image: {}".format(encoding['image'][0].shape))
    exit(0)