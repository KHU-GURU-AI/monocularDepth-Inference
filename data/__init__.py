import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataLoader(args):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(args)
    return data_loader

def CreateRealLoader(args):
    data_loader = RealDatasetDataLoader()
    data_loader.initialize(args)
    return data_loader

def CreateSyntheticLoader(args):
    data_loader = SyntheticDatasetDataLoader()
    data_loader.initialize(args)
    return data_loader

def CreateDataset(args):
    dataset = None
    from data.single_dataset import TestDataset
    dataset = TestDataset()
    print("The dataset has been created")
    dataset.initialize(args)
    return dataset

def CreateRealDataset(args):
    dataset = None
    from data.single_dataset import RealDataset
    dataset = RealDataset()
    print("The real dataset has been created")
    dataset.initialize(args)
    return dataset


def CreateSyntheticDataset(args):
    dataset = None
    from data.single_dataset import SyntheticDataset
    dataset = SyntheticDataset()
    print("The synthetic dataset has been created")
    dataset.initialize(args)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, args):
        BaseDataLoader.initialize(self, args)
        self.dataset = CreateDataset(args)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=32)

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.args.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.args.max_dataset_size:
                break
            yield data


class RealDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'RealDatasetDataLoader'

    def initialize(self, args):
        BaseDataLoader.initialize(self, args)
        self.dataset = CreateRealDataset(args)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=32)

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.args.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.args.max_dataset_size:
                break
            yield data

    def __getitem__(self, index):
        data = self.dataset[index]
        return data



class SyntheticDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'SyntheticDatasetDataLoader'

    def initialize(self, args):
        BaseDataLoader.initialize(self, args)
        self.dataset = CreateSyntheticDataset(args)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=32)

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.args.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.args.max_dataset_size:
                break
            yield data

    def __getitem__(self, index):
        data = self.dataset[index]
        return data