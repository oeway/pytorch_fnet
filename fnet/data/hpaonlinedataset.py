import skimage.external.tifffile as tifffile
import pandas as pd
import numpy as np
import os
import torch
import requests
from PIL import Image

from fnet.data.fnetdataset import FnetDataset


class HPAOnlineDataset(FnetDataset):
    """Dataset for images from the Human Protein Atlas.

    Currently assumes that images are loaded in ZCXY format

    """

    def __init__(self, path_csv: str = None,
                 is_train=True,
                 train_split=0.9,
                 channel_signal=None,
                 channel_target=None,
                 transform_signal=None,
                 transform_target=None):
        path_csv = path_csv or 'https://dl.dropbox.com/s/k9ekd4ff3fyjbfk/umap_results_fit_all_transform_all_sorted_20190422.csv'
        self.df = pd.read_csv(path_csv)
        # filter out invalid rows
        self.df = self.df[self.df['id'].str.contains("_")]
        self.train_split = train_split
        train_split_count = int(len(self.df)*train_split)
        self.is_train = is_train
        if is_train:
            self.df = self.df[:train_split_count]
        else:
            self.df = self.df[train_split_count:]
        
        super().__init__(self.df, None, transform_signal, transform_target)

        self.channel_signal = channel_signal or ['blue', 'red']
        self.channel_target = channel_signal or ['green']
        self.index_dict = {'red': 0, 'green': 1, 'blue': 2}
        self.root_url = 'http://v18.proteinatlas.org/images/'
        self.data_dir = './hpav18-data'
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        assert all(i in self.df.columns for i in ['id'])

    def __getitem__(self, index):
        element = self.df.iloc[index, :]
        has_target = self.channel_target and len(self.channel_target)>0
        img = element['id'].split('_')
        colors = self.channel_signal + self.channel_target
        im_out = []
        for color in colors:
            img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
            img_name = element['id'] + "_" + color + ".jpg"
            img_url = self.root_url + img_path
            file_path = os.path.join(self.data_dir, img_name)
            if not os.path.exists(file_path):
                r = requests.get(img_url, allow_redirects=True)
                open(file_path, 'wb').write(r.content)

            im_tmp = np.array(Image.open(file_path))[:,:,self.index_dict[color]]
            im_out.append(im_tmp)

        if self.transform_signal is not None:
            for t in self.transform_signal:
                for i in range(len(self.channel_signal)):
                    im_out[i] = t(im_out[i])

        offset = len(self.channel_signal)
        if has_target and self.transform_target is not None:
            for t in self.transform_target:
                for i in range(self.channel_target):
                    im_out[offset+i] = t(im_out[offset+i])

        im_out = [torch.from_numpy(im.astype(float)).float() for im in im_out]

        if has_target:
            return torch.stack(im_out[: offset]), torch.stack(im_out[offset:])
        else:
            return torch.stack(im_out[: offset])

    def __len__(self):
        return len(self.df)

    def get_information(self, index: int) -> dict:
        return self.df.iloc[index, :].to_dict()
