"""
	Usages:
		CUDA_VISIBLE_DEVICES=0 python train.py --train_config default_config --work_dir runs/train/layoutlmv2-base-uncased_50e/

"""

import argparse
from transformers import AutoModelForQuestionAnswering
import torch.nn as nn
from utils import create_logger, get_gpu_memory_map, load_feature_from_file
from config import TRAIN_FEATURE_PATH, VAL_FEATURE_PATH, MODEL_CHECKPOINT, TRAINING_CONFIGs
import numpy as np
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(model, train_data, val_data, 
        epochs, optimizer, lr, loss_log, save_freq,
        eval_freq, work_dir, logger):


    optimizer = optimizer(model.parameters(), lr=lr)

    # Measure GPU memory of model
    GPU_usage_before = get_gpu_memory_map()

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        print()
        logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model).cuda()
    model.to(device)
    
    gpus_usage = np.sum(get_gpu_memory_map() - GPU_usage_before)
    logger.info("GPUs usages for model: {} Mb".format(gpus_usage))

    model.train()

    min_valid_loss = np.inf
    idx = 1

    for epoch in range(1, epochs):

        logger.info("Epoch {}/{}".format(epoch, epochs))
        
        train_loss = 0.0
        for _, train_batch in enumerate(train_data):
            input_ids         = train_batch["input_ids"].to(device)
            attention_mask    = train_batch["attention_mask"].to(device)
            token_type_ids    = train_batch["token_type_ids"].to(device)
            bbox              = train_batch["bbox"].to(device)
            image             = train_batch["image"].to(device)
            start_positions   = train_batch["start_positions"].to(device)
            end_positions     = train_batch["end_positions"].to(device)

            # DEBUG for multi-gpus training
            print("Model device: {}".format(model.device_ids))
            print("Weights device: {}".format(next(model.parameters()).device))
            print("input_ids device: {}".format(input_ids.device))
            print("attention_mask device: {}".format(attention_mask.device))
            print("token_type_ids device: {}".format(token_type_ids.device))
            print("bbox device: {}".format(bbox.device))
            print("image device: {}".format(image.device))
            print("start_positions device: {}".format(start_positions.device))
            print("end_positions device: {}".format(end_positions.device))
            input()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
    
            loss = outputs.loss
            
            loss.backward()
            
            optimizer.step()

            train_loss += loss.item()


            # Evaluate current model on entire validation dataset after each `eval_freq` iterations
            if idx % eval_freq == 1:
                val_loss = 0.0
                model.eval()
                for _, val_batch in enumerate(val_data):
                    
                    val_batch               = val_batch.to(device)
                    input_ids               = val_batch["input_ids"].to(device)
                    attention_mask          = val_batch["attention_mask"].to(device)
                    token_type_ids          = val_batch["token_type_ids"].to(device)
                    bbox                    = val_batch["bbox"].to(device)
                    image                   = val_batch["image"].to(device)
                    start_positions         = val_batch["start_positions"].to(device)
                    end_positions           = val_batch["end_positions"].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
                    loss = outputs.loss
                    # Calculate Loss
                    val_loss += loss.item()
                
                logger.info("Iterations: {:<6} - epoch: {:<3} - train_loss: {:<6} - val_loss: {:<6}".format(idx, epoch, train_loss/eval_freq, val_loss/len(val_data)))
                loss_log.info("Iterations: {:<6} - epoch: {:<3} - train_loss: {:<6} - val_loss: {:<6}".format(idx, epoch, train_loss/eval_freq, val_loss/len(val_data)))

                    
                if min_valid_loss > val_loss/len(val_data):
                    logger.info("Found best model !! Validation loss descreased from {} to {}".format(min_valid_loss, val_loss/len(val_data)))
                    torch.save(model.state_dict(), os.path.join(work_dir, 'best'+'.pth'))
                    min_valid_loss = val_loss/len(val_data)

                # Save model each save_freq iteration
                if idx % save_freq == 1:
                    logger.info("Saving model to {}".format(os.path.join(work_dir, str(idx).zfill(5)+'.pth')))
                    torch.save(model.state_dict(), os.path.join(work_dir, str(idx).zfill(5)+'.pth'))

                # Reset training loss
                train_loss = 0.0
                
            idx += 1


    logger.info("Done !")
    logger.info("The minimum on validation {}".format(min_valid_loss))

    return model


def main(args):

    if not os.path.exists(args['work_dir']):
        os.mkdir(args['work_dir'])
    # Create logger
    loss_log = create_logger(os.path.join(args["work_dir"], 'loss.log'))
    logger = create_logger(os.path.join(args['work_dir'], 'log.log'))

    logger.info('Loading training configuration ...')
    config = TRAINING_CONFIGs[args['train_config']]
    optimizer, momentum, lr, epochs, batch_size,\
         eval_freq, save_freq, num_workers = config['optimizer'], config['momentum'], \
        config['lr'], config['epochs'], \
        config['batch_size'], config['eval_freq'], config['save_freq'], config['num_workers']
    logger.info("Configuration: {}".format(config))

    #  Check whether feature path file existing or not
    if not os.path.exists(TRAIN_FEATURE_PATH):
        logger.error("Invalid training feature path")
        exit(0)
    if not os.path.exists(VAL_FEATURE_PATH):
        logger.error("Invalid validation feature path")
        exit(0)

	# Load data into program 
    logger.info("Loading training dataset from {} ...".format(TRAIN_FEATURE_PATH))
    train_dataloader = load_feature_from_file(path=TRAIN_FEATURE_PATH, 
                                            batch_size=batch_size, num_workers=num_workers)

    logger.info("Loading validation dataset from {} ...".format(TRAIN_FEATURE_PATH))
    val_dataloader = load_feature_from_file(path=TRAIN_FEATURE_PATH, 
                                            batch_size=batch_size, num_workers=num_workers)

    logger.info("Training size: {} - Validation size: {}".format(
        len(train_dataloader.dataset), len(val_dataloader.dataset)))

	# Create model for fine-tuning
    logger.info("Loading pre-training model from {} checkpoint".format(MODEL_CHECKPOINT))
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)
    
	# Fine-tuning model
    trained_model = train(model=model, train_data=train_dataloader, val_data=val_dataloader,
						epochs=epochs, optimizer=optimizer, lr=lr, loss_log=loss_log, save_freq=save_freq,
                        work_dir=args['work_dir'], logger=logger, eval_freq=eval_freq)

	


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fine tuning pre-training model on DocVQA data')

    parser.add_argument('--work_dir', default='runs/train/train_1/',
        help='The directory store model checkpoint and log file',
    )

    parser.add_argument('--train_config', default='default_config', 
		help='The training configurations: learning rate, batch size, epochs, optimizer, ...'
	)

    args = vars(parser.parse_args())

    main(args)
