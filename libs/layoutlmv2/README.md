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
*Will coming soon*
### Extracted features
We can found extracted feature at:
	- DocVQA: 
		+ Train
		+ Val
		+ Test
	- InfographicVQA:
		+ Train
		+ Val
		+ Test

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


Contact infomataion: tiennvcs@gmail.com
