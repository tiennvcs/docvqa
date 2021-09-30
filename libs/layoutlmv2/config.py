import os
from datasets import Features, Sequence, Value, Array2D, Array3D
from torch.optim import Adam, SGD


DOCVQA_ROOT_DIR           = '/mlcv/Databases/DocVQA_2020-21/task_1/'
INFOVQA_ROOT_DIR          = '/mlcv/Databases/DocVQA_2020-21/task_3/'

BASE_MODEL_CHECKPOINT     = 'microsoft/layoutlmv2-base-uncased'
LARGE_MODEL_CHECKPOINT    = 'microsoft/layoutlmv2-large-uncased'


features           = Features({
                        'question_id': Value(dtype='int64'),
                        'input_ids': Sequence(feature=Value(dtype='int64')),
                        'bbox': Array2D(dtype="int64", shape=(512, 4)),
                        'attention_mask': Sequence(Value(dtype='int64')),
                        'token_type_ids': Sequence(Value(dtype='int64')),
                        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
                        'start_positions': Value(dtype='int64'),
                        'end_positions': Value(dtype='int64'),
                    })


TRAINING_CONFIGs   = {
    'docvqa_base_sgd': {
        'TRAIN_FEATURE_PATH': os.path.join(DOCVQA_ROOT_DIR, 'extracted_features/layoutlmv2/train/'),
        'VAL_FEATURE_PATH': os.path.join(DOCVQA_ROOT_DIR, 'extracted_features/layoutlmv2/val/'),
        'MODEL'             : BASE_MODEL_CHECKPOINT,
        'optimizer'         : SGD,
        'lr'                : 1e-4,
        'momentum'          : 0.9,
        'epochs'            : 10,
        'batch_size'        : 2,
        'eval_freq'         : 100,
        'save_freq'         : 10000,
        'num_workers'       : 4,
    },

    'infovqa_base_adam': {
        'TRAIN_FEATURE_PATH': os.path.join(INFOVQA_ROOT_DIR, 'extracted_features/layoutlmv2/train/'),
        'VAL_FEATURE_PATH': os.path.join(INFOVQA_ROOT_DIR, 'extracted_features/layoutlmv2/val/'),
        'MODEL'             : BASE_MODEL_CHECKPOINT,
        'optimizer'         : Adam,
        'lr'                : 1e-4,
        'epochs'            : 10,
        'batch_size'        : 2,
        'eval_freq'         : 100,
        'save_freq'         : 10000,
        'num_workers'       : 4,
    },

    'infovqa_base_sgd': {
        'TRAIN_FEATURE_PATH': os.path.join(INFOVQA_ROOT_DIR, 'extracted_features/layoutlmv2/train/'),
        'VAL_FEATURE_PATH': os.path.join(INFOVQA_ROOT_DIR, 'extracted_features/layoutlmv2/val/'),
        'MODEL'             : BASE_MODEL_CHECKPOINT,
        'optimizer'         : SGD,
        'lr'                : 1e-4,
        'momentum'          : 0.9,
        'epochs'            : 10,
        'batch_size'        : 2,
        'eval_freq'         : 100,
        'save_freq'         : 10000,
        'num_workers'       : 4,
    },
    
    'infovqa_base_sgd_2': {
        'TRAIN_FEATURE_PATH': os.path.join(INFOVQA_ROOT_DIR, 'extracted_features/layoutlmv2/train/'),
        'VAL_FEATURE_PATH': os.path.join(INFOVQA_ROOT_DIR, 'extracted_features/layoutlmv2/val/'),
        'MODEL'             : BASE_MODEL_CHECKPOINT,
        'optimizer'         : SGD,
        'lr'                : 2e-5,
        'momentum'          : 0.9,
        'epochs'            : 10,
        'batch_size'        : 2,
        'eval_freq'         : 100,
        'save_freq'         : 10000,
        'num_workers'       : 4,
    },
}


BATCH_SIZE         = 16
DEBUG              = 1000
