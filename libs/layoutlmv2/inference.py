import torch
import argparse
import os
from utils import load_feature_from_file, create_logger
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer



def inference(model, data, tokenizer, output_dir):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()

    for _, val_batch in enumerate(data):
        input_ids               = val_batch["input_ids"].to(device)
        attention_mask          = val_batch["attention_mask"].to(device)
        token_type_ids          = val_batch["token_type_ids"].to(device)
        bbox                    = val_batch["bbox"].to(device)
        image                   = val_batch["image"].to(device)
        start_positions         = val_batch["start_positions"].to(device)
        end_positions           = val_batch["end_positions"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
        start_logits = outputs.start_logits
        end_logits   = outputs.end_logits
        start_pos    = torch.argmax(start_logits)
        end_pos      = torch.argmax(end_logits)

        print(start_pos)
        print(end_pos)

        input()
        


def main(args):

    args['output_dir'] = os.path.join(args['output_dir'], args['weights'].split("/")[-2])
    print(args['output_dir'])
    if not os.path.exists(args['output_dir']):
        try:
            os.mkdir(args['output_dir'])
        except:
            print("INVALID OUTPUT DIRECTORY")
            exit(0)
    
    logger = create_logger(file_path=os.path.join(args['output_dir'], 'inference.log'))

    # Load dataset
    logger.info("Loading dataset from {} ...".format(args['input_dir']))
    eval_data = load_feature_from_file(path=args['input_dir'])
    logger.info("The number of sample is {} ...".format(len(eval_data.dataset)))

    # Load model
    logger.info("Loading model from {} ...".format(args['weights']))
    model = AutoModelForQuestionAnswering.from_pretrained(args['model']).cuda()
    model.load_state_dict(torch.load(args['weights'], map_location='cuda'))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutlmv2-base-uncased')

    # Inference
    logger.info("Start inference ...")
    inference(model=model, data=eval_data, tokenizer=tokenizer, output_dir=args['output_dir'])



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