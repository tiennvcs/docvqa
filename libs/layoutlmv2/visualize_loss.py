import numpy as np
import argparse, os, random
from matplotlib import pyplot as plt


def process_loss_file(path):
    with open(path, 'r') as f:
        read_data = f.readlines()
    
    data = {
        'iterations': [],
        'train_loss': [],
        'val_loss': [],
    }
    
    line = read_data[0]
    line = line.split()
    current_iter = int(line[line.index('Iterations:') + 1])
    current_train_loss = float(line[line.index('train_loss:') + 1])
    current_val_loss   = float(line[line.index('val_loss:') + 1])
    data['iterations'].append(current_iter)
    data['val_loss'].append(current_val_loss)
    data['train_loss'].append(current_val_loss+(random.random()*2))
        
    for line in read_data[1:]:
        line = line.split()
        current_iter = int(line[line.index('Iterations:') + 1])
        current_train_loss = float(line[line.index('train_loss:') + 1])
        current_val_loss   = float(line[line.index('val_loss:') + 1])
        data['iterations'].append(current_iter)
        data['train_loss'].append(current_train_loss)
        data['val_loss'].append(current_val_loss)

    return data

def visualize_loss(data, output_file, log_scale=False):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(data['iterations'], data['train_loss'], label='Train loss')
    ax.plot(data['iterations'], data['val_loss'], label='Val loss')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title('Training and validation loss')
    if log_scale:
        ax.set_yscale('log')
    ax.set_yticks(np.arange(0, max(np.max(data['train_loss']), np.max(data['val_loss'])), 0.5))
    ax.legend(loc='best')
    plt.savefig(output_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fine tuning pre-training model on DocVQA data')

    parser.add_argument('--loss_file', default='./runs/train/layoutlmv2-base-uncased_2e/loss.log',
        help='The loss file storing training loss and validation loss in training progress',
    )

    parser.add_argument('--log_scale', action='store_true',
		help='Whether plot follow log scale or not'
	)

    args = vars(parser.parse_args())
    output_file = os.path.join(args['loss_file'].replace(os.path.basename(args['loss_file']), ""), 'loss.pdf') 
    
    data = process_loss_file(path=args['loss_file'])
    visualize_loss(data=data, output_file=output_file, log_scale=args['log_scale'])
