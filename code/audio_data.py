import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
import librosa as lr
import bisect

import s3fs
import boto3


class WavenetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_file,
                 item_length,
                 target_length,
                 s3_bucket,
                 s3_folder,
                 dataset_name,
                 classes=256,
                 sampling_rate=16000,
                 mono=True,
                 normalize=False,
                 dtype=np.uint8,
                 train=True,
                 test_stride=100):

        #           |----receptive_field----|
        #                                 |--output_length--|
        # example:  | | | | | | | | | | | | | | | | | | | | |
        # target:                           | | | | | | | | | |
        
        # Enables data to be read from and written to s3 buckets
        # s3://bensandboxbucket/WavenetSampleGen/data/{dataset-name}/
        self._fs = s3fs.S3FileSystem()
        self._dataset_name = dataset_name
        self._s3_bucket = s3_bucket
        self._s3_base_folder = s3_folder
        self._s3_data_location = 's3://' + self._s3_bucket + '/' + self._s3_base_folder + '/' + self._dataset_name + '/audio-files/'
        
        # make this /tmp/data.npz
        self.dataset_file = dataset_file
        self._item_length = item_length
        self._test_stride = test_stride
        self.target_length = target_length
        self.classes = classes

        if not os.path.isfile(dataset_file):
            # assert file_location is not None, "no location for dataset files specified"
            self.mono = mono
            self.normalize = normalize

            self.sampling_rate = sampling_rate
            self.dtype = dtype
            self.create_dataset(dataset_file)
        else:
            # Unknown parameters of the stored dataset
            # TODO Can these parameters be stored, too?
            self.mono = None
            self.normalize = None

            self.sampling_rate = None
            self.dtype = None

        self.data = np.load(self.dataset_file, mmap_mode='r')
        self.start_samples = [0]
        self._length = 0
        self.calculate_length()
        self.train = train
        print("one hot input")
        # assign every *test_stride*th item to the test set
    
    def list_all_audio_files(self):
        audio_files = []
        for dirpath, dirnames, filenames in self._fs.walk(self._s3_data_location):
            for filename in [f for f in filenames if f.endswith((".mp3", ".wav", ".aif", "aiff"))]:
                audio_files.append(filename)

        if len(audio_files) == 0:
            print("found no audio files in " + location)
        return audio_files
    
    
    # creates a dataset from files at s3://{self._s3_bucket}/{self._s3_base_folder}/{self.dataset_name}/
    def create_dataset(self, out_file):
        print("create dataset from audio files at", self._s3_data_location)
        self.dataset_file = out_file
        files = self.list_all_audio_files()
        processed_files = []
        s3 = boto3.client('s3')
        for i, file in enumerate(files):
            print("  processed " + str(i) + " of " + str(len(files)) + " files")
            
            # pull one file at a time from s3 bucket
            s3_file = self._s3_base_folder + '/' + self._dataset_name + '/audio-files/' + file
            local_file = '/tmp/dummy.' + file.split('.')[1]
            s3.download_file(self._s3_bucket, s3_file, local_file)
            
            file_data, _ = lr.load(path=local_file,
                                   sr=self.sampling_rate,
                                   mono=self.mono)
            if self.normalize:
                file_data = lr.util.normalize(file_data)
            quantized_data = quantize_data(file_data, self.classes).astype(self.dtype)
            processed_files.append(quantized_data)

        print(files)
        np.savez(self.dataset_file, *processed_files)

    def calculate_length(self):
        start_samples = [0]
        print(self.data.keys())
        for i in range(len(self.data.keys())):
            print(len(self.data['arr_' + str(i)]))
            start_samples.append(start_samples[-1] + len(self.data['arr_' + str(i)]))
        available_length = start_samples[-1] - (self._item_length - (self.target_length - 1)) - 1
        self._length = math.floor(available_length / self.target_length)
        self.start_samples = start_samples

    def set_item_length(self, l):
        self._item_length = l
        self.calculate_length()

    def __getitem__(self, idx):
        if self._test_stride < 2:
            sample_index = idx * self.target_length
        elif self.train:
            sample_index = idx * self.target_length + math.floor(idx / (self._test_stride-1))
        else:
            sample_index = self._test_stride * (idx+1) - 1

        file_index = bisect.bisect_left(self.start_samples, sample_index) - 1
        if file_index < 0:
            file_index = 0
        if file_index + 1 >= len(self.start_samples):
            print("error: sample index " + str(sample_index) + " is to high. Results in file_index " + str(file_index))
        position_in_file = sample_index - self.start_samples[file_index]
        end_position_in_next_file = sample_index + self._item_length + 1 - self.start_samples[file_index + 1]

        if end_position_in_next_file < 0:
            file_name = 'arr_' + str(file_index)
            this_file = np.load(self.dataset_file, mmap_mode='r')[file_name]
            sample = this_file[position_in_file:position_in_file + self._item_length + 1]
        else:
            # load from two files
            file1 = np.load(self.dataset_file, mmap_mode='r')['arr_' + str(file_index)]
            file2 = np.load(self.dataset_file, mmap_mode='r')['arr_' + str(file_index + 1)]
            sample1 = file1[position_in_file:]
            sample2 = file2[:end_position_in_next_file]
            sample = np.concatenate((sample1, sample2))

        example = torch.from_numpy(sample).type(torch.LongTensor)
        one_hot = torch.FloatTensor(self.classes, self._item_length).zero_()
        one_hot.scatter_(0, example[:self._item_length].unsqueeze(0), 1.)
        target = example[-self.target_length:].unsqueeze(0)
        return one_hot, target

    def __len__(self):
        test_length = math.floor(self._length / self._test_stride)
        if self.train:
            return self._length - test_length
        else:
            return test_length


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(mu_x, bins) - 1
    return quantized


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s

