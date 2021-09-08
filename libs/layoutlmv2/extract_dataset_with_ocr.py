import pickle, json, os
import pandas as pd
from datasets import Dataset
from utils import get_ocr_words_and_boxes_train, get_ocr_words_and_boxes_val


def extract_feature_ocr(data_dir, batch_size):
    
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
        output_file = os.path.join(data_dir, 'train_extract_dataset_with_ocr.pk')
        dataset_with_ocr = dataset.map(get_ocr_words_and_boxes_train, batched=True, batch_size=batch_size)
    elif 'val' in data_dir:
        output_file = os.path.join(data_dir, 'val_extract_dataset_with_ocr.pk')
        dataset_with_ocr = dataset.map(get_ocr_words_and_boxes_val, batched=True, batch_size=batch_size)

    with open(output_file, 'wb') as f:
        pickle.dump(dataset_with_ocr, f)

    print("Saved feature file at {}".format(output_file))
    print("Successfully !")


if __name__ == '__main__':
    
    batch_size = 2
    
    data_dir = '/mlcv/Databases/DocVQA_2020-21/task_1/val'
    extract_feature_ocr(data_dir=data_dir, batch_size=batch_size)

    # data_dir = '/mlcv/Databases/DocVQA_2020-21/task_1/train'
    # extract_feature_ocr(data_dir=data_dir, batch_size=batch_size)