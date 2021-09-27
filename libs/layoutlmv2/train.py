"""
	Usages:
		CUDA_VISIBLE_DEVICES=0 python train.py --train_config default_config --work_dir runs/train/layoutlmv2-base-uncased_50e/

"""

import argparse
from torch import distributed as dist
from transformers import AutoModelForQuestionAnswering
import torch.nn as nn
from utils import create_logger, get_gpu_memory_map, load_feature_from_file, setup, cleanup
from config import TRAIN_FEATURE_PATH, VAL_FEATURE_PATH, MODEL_CHECKPOINT, TRAINING_CONFIGs
import numpy as np
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


def train(rank, model, train_data, val_data, world_size,
        epochs, optimizer, lr, save_freq,
        eval_freq, work_dir):


    device = rank
    print("Running DDP with model parallel example on cuda:{} device".format(rank))
    # logger.info("Running DDP with model parallel example on cuda:{} device".format(rank))

    setup(rank, world_size)
    GPU_usage_before = get_gpu_memory_map()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    gpus_usage = np.sum(get_gpu_memory_map() - GPU_usage_before)
    print("GPUs usages for model: {} Mb".format(gpus_usage))
    # logger.info("GPUs usages for model: {} Mb".format(gpus_usage))
    
    optimizer = optimizer(model.parameters(), lr=lr)

    model.train()

    min_valid_loss = np.inf
    idx = 1

    for epoch in range(1, epochs):

        print("Epoch {}/{}".format(epoch, epochs))
        
        # logger.info("Epoch {}/{}".format(epoch, epochs))
        
        train_loss = 0.0
        for _, train_batch in enumerate(train_data):

            input_ids         = train_batch["input_ids"].to(device)
            attention_mask    = train_batch["attention_mask"].to(device)
            token_type_ids    = train_batch["token_type_ids"].to(device)
            bbox              = train_batch["bbox"].to(device)
            image             = train_batch["image"].to(device)
            start_positions   = train_batch["start_positions"].to(device)
            end_positions     = train_batch["end_positions"].to(device)

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
                    
                    # val_batch               = val_batch.to(device)
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
                
                print("Iterations: {:<6} - epoch: {:<3} - train_loss: {:<6} - val_loss: {:<6}".format(idx, epoch, train_loss/eval_freq, val_loss/len(val_data)))
                print("Iterations: {:<6} - epoch: {:<3} - train_loss: {:<6} - val_loss: {:<6}".format(idx, epoch, train_loss/eval_freq, val_loss/len(val_data)))

                # logger.info("Iterations: {:<6} - epoch: {:<3} - train_loss: {:<6} - val_loss: {:<6}".format(idx, epoch, train_loss/eval_freq, val_loss/len(val_data)))
                # loss_log.info("Iterations: {:<6} - epoch: {:<3} - train_loss: {:<6} - val_loss: {:<6}".format(idx, epoch, train_loss/eval_freq, val_loss/len(val_data)))
                    
                if min_valid_loss > val_loss/len(val_data):
                    print("Found best model !! Validation loss descreased from {} to {}".format(min_valid_loss, val_loss/len(val_data)))
                    # logger.info("Found best model !! Validation loss descreased from {} to {}".format(min_valid_loss, val_loss/len(val_data)))
                    torch.save(model.state_dict(), os.path.join(work_dir, 'best'+'.pth'))
                    min_valid_loss = val_loss/len(val_data)

                # Save model each save_freq iteration
                if idx % save_freq == 1:
                    print("Saving model to {}".format(os.path.join(work_dir, str(idx).zfill(5)+'.pth')))
                    # logger.info("Saving model to {}".format(os.path.join(work_dir, str(idx).zfill(5)+'.pth')))
                    torch.save(model.state_dict(), os.path.join(work_dir, str(idx).zfill(5)+'.pth'))
                    dist.barrier()

                # Reset training loss
                train_loss = 0.0
                
            idx += 1


    # logger.info("Done !")
    # logger.info("The minimum on validation {}".format(min_valid_loss))
    print("DONE !")
    print("The minimum on validation {}".format(min_valid_loss))

    cleanup()

    return model


def main(args):

    gpu_ids = [i for i in range(torch.cuda.device_count())]
    torch.cuda.set_device(gpu_ids[0])
    
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

    # Check whether feature path file existing or not
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

    logger.info("Loading validation dataset from {} ...".format(VAL_FEATURE_PATH))
    val_dataloader = load_feature_from_file(path=VAL_FEATURE_PATH, 
                                            batch_size=batch_size, num_workers=num_workers)

    logger.info("Training size: {} - Validation size: {}".format(
        len(train_dataloader.dataset), len(val_dataloader.dataset)))

    logger.info("Loading pre-training model from {} checkpoint".format(MODEL_CHECKPOINT))
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)
    
	# Fine-tuning model
    # trained_model = train(model=model, train_data=train_dataloader, val_data=val_dataloader,
	# 					epochs=epochs, optimizer=optimizer, lr=lr, loss_log=loss_log, save_freq=save_freq,
    #                     work_dir=args['work_dir'], logger=logger, eval_freq=eval_freq, gpu_ids=gpu_ids)

    mp.spawn(train,
             args=(model, train_dataloader, val_dataloader, len(gpu_ids), 
                   epochs, optimizer, lr, save_freq,
                   eval_freq, args['work_dir']),
             nprocs=len(gpu_ids),
             join=True)


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
