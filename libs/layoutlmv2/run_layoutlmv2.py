"""
	Usages:
		python run_layoutlmv2.py --train_config default

"""

import argparse
from transformers import AutoModelForQuestionAnswering
import torch.nn as nn
from utils import load_and_process_data, create_logger
from config import TRAIN_DATA, VAL_DATA, MODEL_CHECKPOINT, TRAINING_CONFIGs
import numpy as np
import torch
import os


def train(model, train_data, val_data, 
        epochs, optimizer, lr, loss_log, save_freq,
        eval_freq, work_dir, logger):

    optimizer = optimizer(model.parameters(), lr=lr)

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

    model.to(device)

    model.train()

    min_valid_loss = np.inf

    for epoch in range(1, epochs):

        logger.info("Epoch {}/{}".format(epoch, epochs))
        train_loss = 0.0
        for _, train_batch in enumerate(train_data):
            
			# get the inputs;
            input_ids = train_batch["input_ids"].to(device)
            attention_mask = train_batch["attention_mask"].to(device)
            token_type_ids = train_batch["token_type_ids"].to(device)
            bbox = train_batch["bbox"].to(device)
            image = train_batch["image"].to(device)
            start_positions = train_batch["start_positions"].to(device)
            end_positions = train_batch["end_positions"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
    
            loss = outputs.loss
            
            loss.backward()
            
            optimizer.step()

            train_loss += loss.item()

        logger.info("Training loss: {}".format(train_loss/len(train_data)))

        # Evaluate on validation set
        if epoch % eval_freq == 0:
            val_loss = 0.0
            model.eval()
            for _, val_batch in enumerate(val_data):
                # Transfer Data to GPU if available
                input_ids = val_batch["input_ids"].to(device)
                attention_mask = val_batch["attention_mask"].to(device)
                token_type_ids = val_batch["token_type_ids"].to(device)
                bbox = val_batch["bbox"].to(device)
                image = val_batch["image"].to(device)
                start_positions = val_batch["start_positions"].to(device)
                end_positions = val_batch["end_positions"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
                loss = outputs.loss
                # Calculate Loss
                val_loss += loss.item()
            
            loss_log.info("Epochs: {:<6}/{} - train_loss: {:<6} - val_loss: {:<6}".format(epoch, epochs, train_loss/len(train_data), val_loss/len(val_data)))
            logger.info("Validation loss: {}".format(val_loss/len(val_data)))

            if epoch % save_freq == 0:
                logger.info("Saving model to {}".format(os.path.join(work_dir, str(epoch).zfill(5)+'.pth')))
                torch.save(model.state_dict(), os.path.join(work_dir, str(epoch).zfill(5)+'.pth'))
                
            if min_valid_loss > val_loss:
                logger.info("Found best model !! Validation loss descreased from {} to {}".format(min_valid_loss, val_loss))
                torch.save(model.state_dict(), os.path.join(work_dir, 'best'+'.pth'))
                min_valid_loss = val_loss


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
	
	# Load data into program 
    logger.info("Loading training dataset ...")
    train_dataloader = load_and_process_data(data_dir=TRAIN_DATA, 
                num_workers=num_workers, batch_size=batch_size)
    logger.info("Loading validation dataset ...")
    val_dataloader   = load_and_process_data(data_dir=VAL_DATA,
                num_workers=num_workers, batch_size=batch_size)
    logger.info("Training size: {} - Validation size: {}".format(
        len(train_dataloader.dataset), len(val_dataloader.dataset)))

	# Create model for fine-tuning
    logger.info("Loading pre-training model from {} checkpoint".format(MODEL_CHECKPOINT))
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)

	# Fine-tuning model
    os.environ['CUDA_VISIBLE_DEVICES'] = 2
    trained_model = train(model=model, train_data=train_dataloader, val_data=val_dataloader,
						epochs=epochs, optimizer=optimizer, lr=lr, loss_log=loss_log, save_freq=save_freq,
                        work_dir=args['work_dir'], logger=logger, eval_freq=eval_freq)

	# Evaluate model on validation set


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
