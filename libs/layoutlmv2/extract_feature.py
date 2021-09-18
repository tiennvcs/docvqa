import os, json, argparse
import pandas as pd
from utils import (encode_dataset, get_avail_ocr_feature, 
                gather_ocr_file, load_feature_from_file)
from datasets import Dataset
from config import DEBUG, features
import torch
import time


def extract_from_dir(data_dir, output_dir, batch_size):

    print("Loading annotation file ...")
    label_file = None
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            label_file = os.path.join(data_dir, file)
            break
    with open(label_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data['data'])
    df['image'] = [os.path.join(data_dir, img_file) for img_file in df['image']]

    # Check availabel OCR file
    ocr_dir = [subdir for subdir in os.listdir(data_dir) if "ocr" in subdir]
    if 'our_output_file' in df.columns:
        df['ocr_output_file'] = [os.path.join(data_dir, ocr_file) for ocr_file in df['ocr_output_file']]
    else:
        full_ocr_dir = os.path.join(data_dir, ocr_dir[0])
        df['ocr_output_file'] = gather_ocr_file(full_ocr_dir, df['image'])
    
    # TRICKY to handle the case final batch is not enoguh number of samples :)
    NUM_SAMPLE = (df.shape[0] // batch_size) * batch_size
    dataset = Dataset.from_pandas(df.iloc[:NUM_SAMPLE])

    print("Loading OCR ...")
    dataset_with_ocr = dataset.map(get_avail_ocr_feature, batched=True, batch_size=batch_size)
    print("Encoding entire data set ...")
    encoded_dataset = dataset_with_ocr.map(encode_dataset, batched=True, batch_size=batch_size,
					remove_columns=dataset_with_ocr.column_names, features=features)    
    output_file = os.path.join(output_dir, 'extracted_feature' + time.strftime("%Y%m%d-%H%M%S"))
    print("Saving extracted feature to {} ...".format(output_file))
    torch.save(encoded_dataset, output_file)
    print("Successfully !")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract feature from directory and store to file')
    parser.add_argument('--input_dir', required=True,
        help='The directory store documents and ocr results'
    )
    parser.add_argument('--output_dir', required=True,
        help='The directory output contain feature file'
    )
    parser.add_argument('--batch_size', required=False, default=16,
        help='The number of batch size when ocr/encoding data'
    )
    args = vars(parser.parse_args())
    extract_from_dir(data_dir=args['input_dir'], output_dir=args['output_dir'])
    dataloader = load_feature_from_file(path=os.path.join(args['output_dir'], 'extracted_feature.pt'), 
                                batch_size=1, num_workers=2)
    print(dataloader)