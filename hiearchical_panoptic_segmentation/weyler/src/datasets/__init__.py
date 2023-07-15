from datasets.MyDataset import MyDataset

def get_dataset(name, dataset_opts):
    if name == "mydataset":
        return MyDataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))