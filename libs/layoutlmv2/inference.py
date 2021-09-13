import torch
import pandas as pd
from PIL import Image
from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D
from transformers import (
    LayoutLMv2FeatureExtractor, AutoModelForQuestionAnswering, 
    LayoutLMv2Processor, AutoTokenizer)
from utils import (get_ocr_words_and_boxes, encode_dataset)
from config import features

data = [

    {
        # 'questionId': 49153, 
        'question': 'What is region number?', 
        'image': '/content/test_sample/test_img.png', 
        # 'docId': 14465, 
        # 'ucsf_document_id': 'pybv0228', 
        # 'ucsf_document_page_no': '81', 
        'answers': ['2000'], 
        # 'data_split': 'val'
    }
]

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)
    
tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutlmv2-base-uncased')
dataset_with_ocr = dataset.map(get_ocr_words_and_boxes, batched=True, batch_size=1)
encoded_dataset = dataset_with_ocr.map(encode_dataset, batched=True, batch_size=1, 
                                       remove_columns=dataset_with_ocr.column_names,
                                       features=features)
encoded_dataset.set_format(type="torch")
dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=1)

## Load model to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForQuestionAnswering.from_pretrained('microsoft/layoutlmv2-base-uncased')
model.to(device)
model.eval()

for idx, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    bbox = batch["bbox"].to(device)
    image = batch["image"].to(device)
    start_positions = batch["start_positions"].to(device)
    end_positions = batch["end_positions"].to(device)

    # forward + backward + optimize
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
    start_position = torch.argmax(outputs.start_logits).cpu().numpy()
    end_position   = torch.argmax(outputs.end_logits).cpu().numpy()
    encoding = tokenizer(dataset_with_ocr['question'], dataset_with_ocr['words'], dataset_with_ocr['boxes'], 
                     max_length=512, padding="max_length", truncation=True)
    answer_pred = tokenizer.decode(encoded_dataset['input_ids'][0][start_position: end_position+1])
    print(answer_pred)
