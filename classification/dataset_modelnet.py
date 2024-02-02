import os
import torch
from torch.utils.data import Dataset, DataLoader
from pyntcloud import PyntCloud

class ModelNet_pointcloud(Dataset):
    def __init__(self, dataset_path, mode= 'train', category= None, transform= None, rate =1.0):
        super().__init__()
        self.transform = transform

        categories_all = [name for name in os.listdir(dataset_path) if name not in ['.DS_Store', 'README.txt']]
        if category:
            if category not in categories_all:
                raise Exception('Categoty not found !')
            self.categories = [category]
        else:
            self.categories = categories_all
        
        self.categories.sort()
        self.path_label_pairs = []
        for category in self.categories:
            label = categories_all.index(category)
            folder_path = os.path.join(dataset_path, category, mode)
            pairs = [(os.path.join(folder_path, name), label) for name in os.listdir(folder_path) if name != '.DS_Store']
            self.path_label_pairs += pairs

            # if(mode== 'train'):
            #     label = categories_all.index(category)
            #     dataset_path='/home/yaoliu/scratch/data/modelnet/ModelNet40SampleNorm2/'
            #     folder_path = os.path.join(dataset_path, category, mode)
            #     pairs = [(os.path.join(folder_path, name), label) for name in os.listdir(folder_path) if name != '.DS_Store']
            #     self.path_label_pairs += pairs
            # pairs.sort()
            # self.path_label_pairs += pairs[0:int(len(pairs)* rate )]
            """
            [('/Users/lanni/Documents/Jupyter/3dgcn/classification/ModelNet40Sample/airplane/train/airplane_0001.ply',
              0),
             ('/Users/lanni/Documents/Jupyter/3dgcn/classification/ModelNet40Sample/airplane/train/airplane_0002.ply',
              0),
            """
            


    def __len__(self):
        return len(self.path_label_pairs)

    def __getitem__(self, index):
        path, label = self.path_label_pairs[index]
        label = torch.LongTensor([label])
        obj = PyntCloud.from_file(path)
        points = torch.FloatTensor(obj.xyz)
        if self.transform:
            points = self.transform(points)
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