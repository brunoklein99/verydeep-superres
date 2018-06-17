import torch.utils.data as data
import torch
import h5py


class Dataset(data.Dataset):
    def __init__(self, file_path):
        super(Dataset, self).__init__()
        hf = h5py.File(file_path)
        self.x = hf.get('data')
        self.y = hf.get('label')

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i, :, :, :]).float()
        y = torch.from_numpy(self.y[i, :, :, :]).float()
        return x, y

    def __len__(self):
        (l, _, _, _) = self.x.shape
        return l
