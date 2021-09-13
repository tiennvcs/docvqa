import torch
import os
import json
import logging
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import LayoutLMv2FeatureExtractor
from config import MODEL_CHECKPOINT, TRAIN_DATA, VAL_DATA, features, DEBUG


def subfinder(words_list, answer_list):
    matches = []
    start_indices = []
    end_indices = []
    for idx, i in enumerate(range(len(words_list))):

        if words_list[i] == answer_list[0] and words_list[i:i + len(answer_list)] == answer_list:
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    if matches:
        return matches[0], start_indices[0], end_indices[0]
    else:
        return None, 0, 0


def get_ocr_words_and_boxes_train(examples):
    feature_extractor = LayoutLMv2FeatureExtractor()

    # get a batch of document images
    images = [Image.open(TRAIN_DATA + image_file).convert("RGB") for image_file in examples['image']]

    # resize every image to 224x224 + apply tesseract to get words + normalized boxes
    encoded_inputs = feature_extractor(images)

    examples['image'] = encoded_inputs.pixel_values
    examples['words'] = encoded_inputs.words
    examples['boxes'] = encoded_inputs.boxes

    return examples


def get_ocr_words_and_boxes_val(examples):
    feature_extractor = LayoutLMv2FeatureExtractor()

    # get a batch of document images
    images = [Image.open(VAL_DATA + image_file).convert("RGB") for image_file in examples['image']]

    # resize every image to 224x224 + apply tesseract to get words + normalized boxes
    encoded_inputs = feature_extractor(images)

    examples['image'] = encoded_inputs.pixel_values
    examples['words'] = encoded_inputs.words
    examples['boxes'] = encoded_inputs.boxes

    return examples


def encode_dataset(examples, max_length=512):
    # take a batch
    questions = examples['question']
    words = examples['words']
    boxes = examples['boxes']

    # encode it
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)

    # next, add start_positions and end_positions
    start_positions = []
    end_positions = []
    answers = examples['answers']
    # for every example in the batch:
    for batch_index in range(len(answers)):
        # print("Batch index:", batch_index)
        cls_index = encoding.input_ids[batch_index].index(tokenizer.cls_token_id)
        # try to find one of the answers in the context, return first match
        words_example = [word.lower() for word in words[batch_index]]
        for answer in answers[batch_index]:
            match, word_idx_start, word_idx_end = subfinder(words_example, answer.lower().split())
            if match:
                break
        # EXPERIMENT (to account for when OCR context and answer don't perfectly match):
        if not match:
            for answer in answers[batch_index]:
                for i in range(len(answer)):
                    # drop the ith character from the answer
                    answer_i = answer[:i] + answer[i + 1:]
                    # check if we can find this one in the context
                    match, word_idx_start, word_idx_end = subfinder(words_example, answer_i.lower().split())
                    if match:
                        break
        # END OF EXPERIMENT

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

            word_ids = encoding.word_ids(batch_index)[token_start_index:token_end_index + 1]
            for id in word_ids:
                if id == word_idx_start:
                    start_positions.append(token_start_index)
                    break
                else:
                    token_start_index += 1

            for id in word_ids[::-1]:
                if id == word_idx_end:
                    end_positions.append(token_end_index)
                    break
                else:
                    token_end_index -= 1
        else:
            start_positions.append(cls_index)
            end_positions.append(cls_index)

    encoding['image'] = examples['image']
    encoding['start_positions'] = start_positions
    encoding['end_positions'] = end_positions

    return encoding


def load_and_process_data(data_dir, batch_size, num_workers):
    label_file = None
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            label_file = os.path.join(data_dir, file)
            break
    with open(label_file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data['data'])
    dataset = Dataset.from_pandas(df.iloc[:DEBUG])
    if 'train' in data_dir:
        dataset_with_ocr = dataset.map(get_ocr_words_and_boxes_train, batched=True, batch_size=batch_size)
    elif 'val' in data_dir:
        dataset_with_ocr = dataset.map(get_ocr_words_and_boxes_val, batched=True, batch_size=batch_size)
    encoded_dataset = dataset_with_ocr.map(encode_dataset, batched=True, batch_size=batch_size,
					remove_columns=dataset_with_ocr.column_names, features=features)
    encoded_dataset.set_format(type="torch")
    dataloader = torch.utils.data.DataLoader(encoded_dataset, shuffle=True,
                                            batch_size=batch_size, num_workers=num_workers)
    return dataloader


def create_logger(file_path):
    log = logging.getLogger(file_path)
    log.setLevel(logging.DEBUG)

    if 'loss' in file_path:
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
