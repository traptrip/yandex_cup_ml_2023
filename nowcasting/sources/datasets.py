import h5py
import numpy as np
import torch.utils.data as data


class RadarDataset(data.Dataset):

    def __init__(self, list_of_files, in_seq_len=4, out_seq_len=12, mode='overlap', with_time=False):
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.seq_len = in_seq_len + out_seq_len
        self.with_time = with_time
        self.__prepare_timestamps_mapping(list_of_files)
        self.__prepare_sequences(mode)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        data = []
        for timestamp in self.sequences[index]:
            with h5py.File(self.timestamp_to_file[timestamp]) as d:
                data.append(np.array(d[timestamp]['intensity']))
        data = np.expand_dims(data, axis=1)
        data[data == -1e6] = 0
        data[data == -2e6] = -1
        inputs = data[:self.in_seq_len]
        targets = data[self.in_seq_len:]
        if self.with_time:
            return (inputs, self.sequences[index][-1]), targets
        else:
            return inputs, targets

    def __prepare_timestamps_mapping(self, list_of_files):
        self.timestamp_to_file = {}
        for filename in list_of_files:
            with h5py.File(filename) as d:
                self.timestamp_to_file = {
                    **self.timestamp_to_file,
                    **dict(map(lambda x: (x, filename), d.keys()))
                }

    def __prepare_sequences(self, mode):
        timestamps = np.unique(sorted(self.timestamp_to_file.keys()))
        if mode == 'sequentially':
            self.sequences = [
                timestamps[index * self.seq_len: (index + 1) * self.seq_len]
                for index in range(len(timestamps) // self.seq_len)
            ]
        elif mode == 'overlap':
            self.sequences = [
                timestamps[index: index + self.seq_len]
                for index in range(len(timestamps) - self.seq_len + 1)
            ]
        else:
            raise Exception(f'Unknown mode {mode}')
        self.sequences = list(filter(
            lambda x: int(x[-1]) - int(x[0]) == (self.seq_len - 1) * 600,
            self.sequences
        ))
