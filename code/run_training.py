import time
from scipy.io import wavfile

# local files
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *


dtype = torch.FloatTensor
ltype = torch.LongTensor

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

model = WaveNetModel(layers=10,
                     blocks=3,
                     dilation_channels=32,
                     residual_channels=32,
                     skip_channels=1024,
                     end_channels=512,
                     output_length=8,
                     dtype=dtype,
                     bias=True)

#model = load_latest_model_from('snapshots', use_cuda=True)
#model = torch.load('snapshots/some_model')

if use_cuda:
    print("move model to gpu")
    model.cuda()

print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

data = WavenetDataset(dataset_file='/tmp/experiment/dataset.npz',
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
                      s3_bucket='bensandboxbucket',
                      s3_folder='WavenetSampleGen/data',
                      dataset_name='basic-jazz',
                      test_stride=500)
print('the dataset has ' + str(len(data)) + ' items')

trainer = WavenetTrainer(model=model,
                         dataset=data,
                         s3_folder='WavenetSampleGen/',
                         s3_bucket='bensandboxbucket',
                         lr=0.0001,
                         weight_decay=0.0,
                         snapshot_interval=1000,
                         dtype=dtype,
                         ltype=ltype)

print('start training...')
trainer.train(batch_size=16,
              epochs=10,
              continue_training_at_step=0)
