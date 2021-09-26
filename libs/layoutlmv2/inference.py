import torch
import argparse
import os, time
"""
Usage:
    CUDA_VISIBLE_DEVICES=0 python inference.py \
        --input_dir /mlcv/Dataset/DocVQA_2020-21/task_1/extracted_features/layoutlmv2/val/ \
        --model microsoft/layoutlmv2-base-uncased --weights ./runs/train/layoutlmv2-base-uncased/best.pth \
        --output_dir ./runs/inference/docvqa/
"""


import json, tqdm
from utils import find_highest_score_answer, load_feature_from_file, create_logger
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer


def inference(model, data, tokenizer):


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()

    results = []

    for _, batch in tqdm.tqdm(enumerate(data)):
        
        input_ids               = batch["input_ids"].to(device)
        attention_mask          = batch["attention_mask"].to(device)
        token_type_ids          = batch["token_type_ids"].to(device)
        bbox                    = batch["bbox"].to(device)
        image                   = batch["image"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    bbox=bbox, image=image)
        start_logits = outputs.start_logits.detach().cpu().numpy()
        end_logits   = outputs.end_logits.detach().cpu().numpy()
        start_indices, end_indices = find_highest_score_answer(
                                    start_scores=start_logits, end_scores=end_logits)
        
        input_ids    = input_ids.cpu().numpy()
        question_ids = batch["question_id"].detach().cpu().numpy().tolist()

        for question_id, input_id, s, e in zip(question_ids, input_ids, start_indices, end_indices):
            predicted_answer = tokenizer.decode(input_id[s:e+1])
            decoding_string  = tokenizer.decode(input_id)
            question         = decoding_string[decoding_string.find('[CLS]')+5:decoding_string.find('[SEP]')]
            results.append({
                "questionId": question_id,
                "question": question,
                "answer": predicted_answer,
            })
    return results


def main(args):

    output_dir = os.path.join(args['output_dir'], args['weights'].split("/")[-2])
    if not os.path.exists(output_dir):
        try:
            print("Creating {} directory".format(output_dir))
            os.mkdir(output_dir)
        except:
            print("INVALID OUTPUT DIRECTORY")
            exit(0)
    
    logger = create_logger(file_path=os.path.join(output_dir, 'inference.log'))

    # Load dataset
    logger.info("Loading dataset from {} ...".format(args['input_dir']))
    eval_data = load_feature_from_file(path=args['input_dir'], batch_size=2)
    logger.info("The number of sample is {} ...".format(len(eval_data.dataset)))

    # Load model
    logger.info("Loading model from {} ...".format(args['weights']))
    model = AutoModelForQuestionAnswering.from_pretrained(args['model']).cuda()
    # model.load_state_dict(torch.load(args['weights'], map_location='cuda'))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args['model'])

    # Inference
    logger.info("Start inference ...")
    start_time  = time.time()
    results     = inference(model=model, data=eval_data, tokenizer=tokenizer)
    end_time    = time.time()
    logger.info("Total inference time {} seconds".format(end_time-start_time))
    
    # save inference results to disk
    output_file = os.path.join(output_dir, args['input_dir'].split("/")[-2] + '.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info("DONE ! Check the inference results at {}".format(output_file))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Inference on DocVQA dataset.')

    parser.add_argument('--input_dir', required=True,
        help='The input feature extracted from DocVQA dataset. I can be train/val/test subset.' ,
    )

    parser.add_argument('--model', default='microsoft/layoutlmv2-base-uncased',
        help='The model architecture.'
    )
    parser.add_argument('--weights', required=True,
        help='The path to model weights',
    )

    parser.add_argument('--output_dir', required=True,
        help='The output directory'
    )

    args = vars(parser.parse_args())

    main(args)
