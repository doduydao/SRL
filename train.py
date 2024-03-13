from data_processor import *
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
# from model.loader import load_sentences, update_tag_scheme, parse_config
# from model.layered_model import Model, Evaluator, Updater


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10

pathdata = "dataset"
viSRLprocessor = Vi_SRL_processor(pathdata, vi_tokenizer)

tokenizer = viSRLprocessor.tokenizer
label2id = viSRLprocessor.label2id
id2label = viSRLprocessor.label2id
train_df, dev_df, test_df = viSRLprocessor.instances

training_set = dataset(train_df, tokenizer, MAX_LEN, label2id)
dev_set = dataset(dev_df, tokenizer, MAX_LEN, label2id)
testing_set = dataset(test_df, tokenizer, MAX_LEN, label2id)



train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }


training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


for batch_X, batch_y in training_loader:
    print(batch_X)
    print(batch_y)
