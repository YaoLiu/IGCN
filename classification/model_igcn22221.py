import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
import igcn as gcn3d
import time
from torchsummary import summary
from thop import profile
from ptflops import get_model_complexity_info


class GCN3D(nn.Module):
    def __init__(self, support_num: int, neighbor_num: int):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(4, 32, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(32, 32, support_num= support_num)
        # self.conv_2 = gcn3d.Conv_layer(64, 32, support_num= support_num)

        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 2, neighbor_num= 4)

        self.conv_3 = gcn3d.Conv_layer(64, 32, support_num= support_num)
        self.conv_4 = gcn3d.Conv_layer(96, 32, support_num= support_num)
        # self.conv_5 = gcn3d.Conv_layer(160, 32, support_num= support_num)

        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 2, neighbor_num= 4)

        self.conv_6 = gcn3d.Conv_layer(128, 32, support_num= support_num)
        self.conv_7 = gcn3d.Conv_layer(160, 32, support_num= support_num)

        self.pool_3 = gcn3d.Pool_layer(pooling_rate= 2, neighbor_num= 4)

        self.conv_8 = gcn3d.Conv_layer(192, 32, support_num= support_num)
        self.conv_9 = gcn3d.Conv_layer(224, 32, support_num= support_num)

        self.pool_4 = gcn3d.Pool_layer(pooling_rate= 2, neighbor_num= 4)

        self.conv_10 = gcn3d.Conv_layer(256, 32, support_num= support_num)
        self.conv_11 = gcn3d.Conv_layer(288, 1024, support_num= support_num)

        self.classifier = nn.Sequential(
            # nn.Linear(1024, 1024), 
            # nn.Dropout(0.5),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(inplace= True),

            nn.Linear(1024, 256), 
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= True),

            nn.Linear(256, 40)
        )

    def forward(self,  vertices: "(bs, vertice_num, 3)"):
        bs, vertice_num, _ = vertices.size()
        # neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        neighbor_index = gcn3d.query_ball_point(0.2, self.neighbor_num, vertices, vertices)


        #fm_0 = self.conv_0(neighbor_index, vertices, vertices)
        fm_0 = self.conv_0(neighbor_index, vertices, vertices)
        fm_0 = F.relu(fm_0, inplace= True)
        # fm_0 = torch.cat((vertices, fm_0), 2)

        fm_1 = self.conv_1(neighbor_index, vertices, fm_0)
        fm_1 = F.relu(fm_1, inplace= True)
        fm_1 = torch.cat((fm_0, fm_1), 2)

        # fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        # fm_2 = F.relu(fm_2, inplace= True) 
        # fm_2 = torch.cat((fm_1, fm_2), 2)

        vertices, fm_1 = self.pool_1(vertices, fm_1, 0.2)
        # neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        neighbor_index = gcn3d.query_ball_point(0.4, self.neighbor_num, vertices, vertices)

        fm_3 = self.conv_3(neighbor_index, vertices, fm_1)
        fm_3 = F.relu(fm_3, inplace= True) 
        fm_3 = torch.cat((fm_1, fm_3), 2)

        fm_4 = self.conv_4(neighbor_index, vertices, fm_3)
        fm_4 = F.relu(fm_4, inplace= True) 
        fm_4 = torch.cat((fm_3, fm_4), 2)

        # fm_5 = self.conv_5(neighbor_index, vertices, fm_4)
        # fm_5 = F.relu(fm_5, inplace= True) 
        # fm_5 = torch.cat((fm_4, fm_5), 2)

        vertices, fm_4 = self.pool_2(vertices, fm_4, 0.4)
        # neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        neighbor_index = gcn3d.query_ball_point(0.6, self.neighbor_num, vertices, vertices)

        fm_6 = self.conv_6(neighbor_index, vertices, fm_4)
        fm_6 = F.relu(fm_6, inplace= True) 
        fm_6 = torch.cat((fm_4, fm_6), 2)

        fm_7 = self.conv_7(neighbor_index, vertices, fm_6)
        fm_7 = F.relu(fm_7, inplace= True) 
        fm_7 = torch.cat((fm_6, fm_7), 2)

        vertices, fm_7 = self.pool_3(vertices, fm_7, 0.6)
        # neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        neighbor_index = gcn3d.query_ball_point(0.8, self.neighbor_num, vertices, vertices)

        fm_8 = self.conv_8(neighbor_index, vertices, fm_7)
        fm_8 = F.relu(fm_8, inplace= True) 
        fm_8 = torch.cat((fm_7, fm_8), 2)

        fm_9 = self.conv_9(neighbor_index, vertices, fm_8)
        fm_9 = F.relu(fm_9, inplace= True) 
        fm_9 = torch.cat((fm_8, fm_9), 2)

        vertices, fm_9 = self.pool_4(vertices, fm_9, 0.8)
        # neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        neighbor_index = gcn3d.query_ball_point(1, self.neighbor_num, vertices, vertices)

        fm_10 = self.conv_10(neighbor_index, vertices, fm_9)
        fm_10 = F.relu(fm_10, inplace= True) 
        fm_10 = torch.cat((fm_9, fm_10), 2)
        
        fm_11 = self.conv_11(neighbor_index, vertices, fm_10)

        feature_global = fm_11.max(1)[0]
        
        pred = self.classifier(feature_global)
        return pred

def test():
    import time
    sys.path.append("..")
    from util import parameter_number
    
    device = torch.device('cuda:0')
    points = torch.zeros(8, 1024, 3).to(device)
    model = GCN3D(support_num= 4, neighbor_num= 32).to(device)
    start = time.time()
    output = model(points)
    
    # print("Inference time: {}".format(time.time() - start))
    print("Parameter #: {}".format(parameter_number(model)))
    # print("Inputs size: {}".format(points.size()))
    # print("Output size: {}".format(output.size()))

# def params():
    # device = torch.device('cuda:2') # PyTorch v0.4.0
    # points = torch.zeros(8, 1024, 3).to(device)
    # model = GCN3D(support_num= 4, neighbor_num= 32).to(device)
    summary(model, (1024, 3))

# def params3():
#     model = GCN3D()
#     input = torch.randn(1, 3, 1024)
#     macs, params = profile(model, inputs=(points, ))

    # net = models.densenet161()
    # macs, params = get_model_complexity_info(model, (8, 1024, 3), as_strings=True,
    #                                     print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
    test()