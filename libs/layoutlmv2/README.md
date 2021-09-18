# LayoutLMv2 implementation for Document visual question answering

## Quick start

- Fine tuning: [colab notebook](https://colab.research.google.com/drive/1uzNjnzBDyRGVgvAbZHT6FS9Ismj-CE40?usp=sharing)
- Inference: [colab notebook](https://colab.research.google.com/drive/1JowmcyoKvxdAblBf6hzVcUiQJQ9G_uEK?usp=sharing)


## Installation

```bash
$ git clone https://github.com/tiennvcs/docvqa
```
```bash
$ cd docvqa/libs/layoutlmv2/
```
```bash
$ chmod 777 ./install.sh
```
```bash
$ ./install.sh
```

## Preparation

### Dataset
The source code implement handle the DocVQA dataset. If you want to running on your own dataset, it must have structure like bellow.
```
/Data/root/directory/
|-- extracted_features
|   |-- test
|   |-- train
|   |-- val
|-- test
|   |-- documents
|   |-- ocr_results
|   `-- test_v1.0.json
|-- train
|   |-- documents
|   |-- ocr_results
|   `-- train_v1.0.json
|-- val
|   |-- documents
|   |-- ocr_results
|   `-- val_v1.0.json
```
#### Annotation
The annotation files (`{train_v1.0/val_v1.0/test.v1.0}.json`) must have the same and following format.

```json
{
    "dataset_name": "docvqa",
    "dataset_version": "1.0",
    "dataset_split": "val",
    "data": [
        {
            "questionId": 49153,
            "question": "What is the ‘actual’ value per 1000, during the year 1975?",
            "image": "documents/pybv0228_81.png",
            "docId": 14465,
            "ucsf_document_id": "pybv0228",
            "ucsf_document_page_no": "81",
            "answers": [
                "0.28"
            ],
            "data_split": "val"
        },
        {
            "questionId": 24580,
            "question": "What is name of university?",
            "image": "documents/nkbl0226_1.png",
            "docId": 7027,
            "ucsf_document_id": "nkbl0226",
            "ucsf_document_page_no": "1",
            "answers": [
                "university of california",
                "University of California",
                "university of california, san diego"
            ],
            "data_split": "val"
        },
	...
    ]
}
```

#### OCR file

All OCR file must be putted in the `ocr_results` folder. Each file must have the same structure like [`ocr_samle.json`]('./sample/ocr_sample.json').

*If your dataset don't have availabel OCR files, don't worry. When running the extract feature section, the Tesseract engine will OCR the images and gather OCR information to feature file.*

### Download 

Here is some availabel dataset containing extracted features and also converted structure.

- Dataset
	- DocVQA dataset *(coming soon)*
	- InfographicVQA dataset *(coming soon)*
	- VietInfographicVQA dataset *(coming soon)*
- Model
        - Our model fine-tuning in 2 epoch on DocVQA dataset *(coming soon)*
        - Our model fine-tuning in 2 epoch on InfographicVQA dataset *(coming soon)*
        - Our model fine-tuning in 2 epoch on InfographicVQA dataset *(coming soon)*


### Extracted features

If you want to fine-tuning pre-training model on your own dataset. You need to extract feature from its. A sample feature is define a dictionay bellow:

```python
features           = Features({
                        'input_ids': Sequence(feature=Value(dtype='int64')),
                        'bbox': Array2D(dtype="int64", shape=(512, 4)),
                        'attention_mask': Sequence(Value(dtype='int64')),
                        'token_type_ids': Sequence(Value(dtype='int64')),
                        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
                        'start_positions': Value(dtype='int64'),
                        'end_positions': Value(dtype='int64'),
                    })
```

Run the following command to extract feature from validation dataset directory (train and test dataset will be performed similar).
```bash
	python extract_feature.py \
		    --input_dir /mlcv/Databases/DocVQA_2020-21/task_1/val/
		    --output_dir /mlcv/Databases/DocVQA_2020-21/task_1/extracted_features/val/
		    --batch_size 16
```
If run successfully, check the output feature file at `/mlcv/Databases/DocVQA_2020-21/task_1/extracted_features/val`.


### Fine-tuned model
	- Model 1
	- Model 2
	
## Usage
### Fine tuning model on DocVQA dataset

**Fine tuning on DocVQA**


**Fine tuning on Infographic VQA**


### Inference

Given an input image and a natural language question, model need to output the natural language words for answering.

1. Create a json file for query. It is a list contain a lot of dictionaries as bellow.
```python
[
	{
		"question": "Which social platform has heavy female audience?", 
		"image": "37313.jpeg",
		"ocr": "37313.json",
	},
  	...
]
```

3. Run the model given json query

```bash
$ python inference.py --input input.json --model_type microsoft/layoutlmv2-base-uncased --weights path/to/fine-tuned-model/
```

*Example:*
```bash
$ python inference.py --input example_input.json --model_type microsoft/layoutlmv2-base-uncased --weights path/to/fine-tuned-model/
```

*Output:*
```bash
AHIHI
```


## Experiment and Results
*Will coming soon*
### Fine-tuning plot


### Summary results

Contact infomataion: tiennvcs@gmail.com
