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

Run the following command to extract feature from train/val/test directory
```bash

```

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
