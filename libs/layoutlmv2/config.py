import os
from datasets import Features, Sequence, Value, Array2D, Array3D
from torch.optim import AdamW

ROOT_DATA  = '/mlcv/Databases/DocVQA_2020-21/task_1/'
TRAIN_DATA = os.path.join(ROOT_DATA, 'train/')
VAL_DATA   = os.path.join(ROOT_DATA, 'val/')
MODEL_CHECKPOINT = 'microsoft/layoutlmv2-base-uncased'


features = Features({
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'start_positions': Value(dtype='int64'),
    'end_positions': Value(dtype='int64'),
})


TRAINING_CONFIGs = {
    'default_config': {
                            'optimizer': AdamW,
                            'lr': 1e-3,
                            'epochs': 50,
                            'batch_size': 2,
                            'momentum': 0.9,
                            'eval_freq': 1,
                            'save_freq': 10,
                            'num_workers': 2,
                    },
    
}
