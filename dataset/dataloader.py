import sys
sys.path.append('../')
import os
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa


class AudioData(Dataset):
    def __init__(self, dimension=21000, stride=4096, fs=22050, scale=2, path="./wav", singer = "NJAT"):
        super(AudioData, self).__init__()
        self.dimension = dimension
        self.stride = stride
        self.scale = scale
        self.fs = fs
        self.singer = os.listdir(path)
        # self.folder_path = [os.path.join(path, name) for name in self.singer]
        self.data_info = {} # {'NJAT': ['16', '15', '07', '20'],...}
        self.wav_dict = {} # {'NJAT': ['./wav/NJAT/16.wav',...],...}

        for i in self.singer:
            folder = os.path.join(path, i)
            self.data_info[i] = []
            self.wav_dict[i] = []
            for songid in os.listdir(folder):
                self.wav_dict[i].append(os.path.join(folder, songid))
                self.data_info[i].append(songid[:-4])

        self.split_dict = {} # {'NJAT': [frame1, frame2,...],...}
        self.split()

    def split(self):
        for name in self.singer:
            self.split_dict[name] = []
            wav_paths = self.wav_dict[name]
            for wavpath in wav_paths:
                wav, _ = librosa.load(path=wavpath, sr=self.fs)
                wav_length = len(wav)  
                if wav_length < self.stride:  
                    continue
                if wav_length < self.dimension:  
                    diffe = self.dimension - wav_length
                    wav_pad = np.pad(wav, (0, diffe), mode="constant")
                    self.split_dict[name].append(wav_pad)
                else:  
                    start_index = 0
                    while True:
                        if start_index + self.dimension > wav_length:
                            break
                        wav_frame = wav[start_index:start_index + self.dimension]
                        self.split_dict[name].append(wav_frame)
                        start_index += self.stride

    def __len__(self):
        return len(self.singer)
    
    def __getitem__(self, singer_name):
        return self.data_info[singer_name], self.split_dict[singer_name]


class ParallelData(Dataset):
    def __init__(self, dimension=21000, stride=4096, fs=22050, scale=2, path="./wav", singer = "NJAT"):
        super(ParallelData, self).__init__()
        self.dimension = dimension
        self.stride = stride
        self.scale = scale
        self.fs = fs
        self.singer = os.listdir(path)
        # self.folder_path = [os.path.join(path, name) for name in self.singer]
        self.data_info = {} # {'NJAT': ['16', '15', '07', '20'],...}
        self.wav_dict = {} # {'NJAT': ['./wav/NJAT/16.wav',...],...}

        for i in self.singer:
            folder = os.path.join(path, i)
            self.data_info[i] = []
            self.wav_dict[i] = []
            for songid in os.listdir(folder):
                self.wav_dict[i].append(os.path.join(folder, songid))
                self.data_info[i].append(songid[:-4])

        self.split_dict = {} # {'NJAT': {'16':[frame1, frame2,...],...},...}
        self.split()

    def split(self):
        for name in self.singer:
            self.split_dict[name] = {}
            wav_paths = self.wav_dict[name]
            for wavpath in wav_paths:
                songid = wavpath[-6:-4]
                self.split_dict[name][songid] = []
                wav, _ = librosa.load(path=wavpath, sr=self.fs)
                wav_length = len(wav)  
                if wav_length < self.stride:  
                    continue
                if wav_length < self.dimension:  
                    diffe = self.dimension - wav_length
                    wav_pad = np.pad(wav, (0, diffe), mode="constant")
                    self.split_dict[name][songid].append(wav_pad)
                else:  
                    start_index = 0
                    while True:
                        if start_index + self.dimension > wav_length:
                            break
                        wav_frame = wav[start_index:start_index + self.dimension]
                        self.split_dict[name][songid].append(wav_frame)
                        start_index += self.stride

    def __len__(self):
        return len(self.singer)
    
    def __getitem__(self, singer_name):
        return self.data_info[singer_name], self.split_dict[singer_name]


# default test case
def main():
    start_time = time.time()
    # data = AudioData()
    data = ParallelData()
    mydatainfo, singersplit = data.__getitem__("ADIZ")
    mydatasplit = singersplit["01"]

    train_loader = DataLoader(mydatasplit, batch_size=32, shuffle=True, drop_last=True)
    end_time = time.time()
    print("spent: %s" % (end_time-start_time), "seconds")    # 88.3641 seconds


if __name__ == "__main__":
    main()