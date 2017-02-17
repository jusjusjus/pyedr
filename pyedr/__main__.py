
from .database import Dataset

#logging.basicConfig(level=logging.DEBUG)
dataset = Dataset(subject_ids=['f1o01'])
data = dataset.get_data_batches(sequence_len=250, sequences_per_batch=12)
feature_batches, target_batches = data
print('feature batches.shape', feature_batches.shape)
print('target batches.shape',  target_batches.shape)
