import argparse

import h5py
import lightning as L
import numpy as np
import torch.utils.data as data
import tqdm

from datasets import RadarDataset
from models import ConvLSTMModel, PersistantModel


def prepare_data_loaders(train_batch_size=6, valid_batch_size=1, test_batch_size=1):
    train_dataset = RadarDataset([
        '../train/2021-01-train.hdf5', '../train/2021-03-train.hdf5', '../train/2021-04-train.hdf5',
        '../train/2021-06-train.hdf5', '../train/2021-07-train.hdf5', '../train/2021-09-train.hdf5',
        '../train/2021-10-train.hdf5', '../train/2021-12-train.hdf5'])
    valid_dataset = RadarDataset([
        '../train/2021-02-train.hdf5', '../train/2021-05-train.hdf5', '../train/2021-08-train.hdf5',
        '../train/2021-11-train.hdf5'
    ])
    test_dataset = RadarDataset(['../2022-test-public.hdf5'], out_seq_len=0, with_time=True)
    train_loader = data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def evaluate_on_val(model, valid_loader):
    rmses = np.zeros((12,), dtype=float)
    for item in tqdm.tqdm(valid_loader):
        inputs, target = item
        output = model(inputs)
        rmses += np.sum((
            np.square(target.detach().numpy() - output.detach().numpy())
        ) * (target.detach().numpy() != -1), axis=(0, 2, 3, 4))
    rmses /= len(valid_loader)
    return np.mean(np.sqrt(rmses))


def process_test(model, test_loader, output_file='../output.hdf5'):
    model.eval()
    for index, item in tqdm.tqdm(enumerate(test_loader)):
        (inputs, last_input_timestamp), _ = item
        output = model(inputs)
        with h5py.File(output_file, mode='a') as f_out:
            for index in range(output.shape[1]):
                timestamp_out = str(int(last_input_timestamp[-1]) + 600 * (index + 1))
                f_out.create_group(timestamp_out)
                f_out[timestamp_out].create_dataset(
                    'intensity',
                    data=output[0, index, 0].detach().numpy()
                )


def main(model_name, tensorboard_path):
    train_loader, valid_loader, test_loader = prepare_data_loaders()
    if model_name == 'persistant':
        # score on valid set: 197.64139689523992
        # score on test set: 283.66210850104176
        model = PersistantModel()
        print(evaluate_on_val(model, valid_loader))
        process_test(model, test_loader)
    elif model_name == 'convlstm':
        model = ConvLSTMModel()
        trainer = L.Trainer(
            logger=L.pytorch.loggers.TensorBoardLogger(save_dir=tensorboard_path),
            max_epochs=1
        )
        trainer.fit(model, train_loader)
        print(evaluate_on_val(model, valid_loader))
        process_test(model, test_loader)
    else:
        print('Unknown model name')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--tensorboard_path', default='/mnt/tensorboard')
    args = parser.parse_args()
    main(args.model, args.tensorboard_path)
