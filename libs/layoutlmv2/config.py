import os, glob2
from datasets import Features, Sequence, Value, Array2D, Array3D
from torch.optim import AdamW, SGD, Adam


ROOT_DIR           = '/mlcv/Databases/DocVQA_2020-21/task_3/'
TRAIN_DIR          = os.path.join(ROOT_DIR, 'train/')
VAL_DIR            = os.path.join(ROOT_DIR, 'val/')
try:
    FEATURE_DIR        = os.path.join(ROOT_DIR, 'extracted_features/layoutlmv2')
    TRAIN_FEATURE_PATH = os.path.join(FEATURE_DIR, 'train')
    VAL_FEATURE_PATH   = os.path.join(FEATURE_DIR, 'val')
except:
    print("DON'T WORRY !")
MODEL_CHECKPOINT   = 'microsoft/layoutlmv2-base-uncased'
BATCH_SIZE         = 16          # Change if want extract feature with your own batch size
DEBUG              = 1000

features           = Features({
                        'input_ids': Sequence(feature=Value(dtype='int64')),
                        'bbox': Array2D(dtype="int64", shape=(512, 4)),
                        'attention_mask': Sequence(Value(dtype='int64')),
                        'token_type_ids': Sequence(Value(dtype='int64')),
                        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
                        'start_positions': Value(dtype='int64'),
                        'end_positions': Value(dtype='int64'),
                    })


TRAINING_CONFIGs   = {
                    'default_config': {
                                            'optimizer': Adam,
                                            'lr': 1e-4,
                                            'epochs': 2,
                                            'batch_size': 2,
                                            'momentum': 0.9,
                                            'eval_freq': 10,       # Evaluate model each 100 iterations
                                            'save_freq': 500,       # Save checkpoint each 500 iterations, it must be divisible for eval_freq
                                            'num_workers': 4,
                                    },                
}
