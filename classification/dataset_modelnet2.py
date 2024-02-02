import os
import torch
from torch.utils.data import Dataset, DataLoader
from pyntcloud import PyntCloud
import pickle

class ModelNet_pointcloud(Dataset):
    def __init__(self, dataset_path, mode= 'train', category= None, transform= None, rate =1.0):
        super().__init__()
        if(mode=='train'):
            f = open('./modelnet40_aug4590.cpkl', "rb")
            self.loadData = pickle.load(f)
            f.close()
        else:
            f = open('./modelnet40_aug4590_test.cpkl', "rb")
            self.loadData = pickle.load(f)
            f.close()
            


    def __len__(self):
        return len(self.loadData)

    def __getitem__(self, index):
        # path, label = self.path_label_pairs[index]
        label = self.loadData[index][0][0]
        # obj = PyntCloud.from_file(path)
        points = self.loadData[index][1][0]
        # if self.transform:
        #     points = self.transform(points)
        return points, label

def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-source')
    parser.add_argument('-bs', type= int, default= 8)
    args = parser.parse_args()
    
    dataset = ModelNet_pointcloud(args.source, mode= "test")
    print("# of Data:", len(dataset))
    dataloader = DataLoader(dataset, batch_size= args.bs)
    for i, (points, labels) in enumerate(dataloader):    
        print(points.size())
        print(labels.size())
        break
        """
        bash-3.2$ python dataset_modelnet.py -source /Users/lanni/Documents/Jupyter/3dgcn/classification/ModelNet40Sample
        # of Data: 2468
        torch.Size([8, 1024, 3])
        torch.Size([8, 1])
        """

if __name__ == '__main__':
    test()