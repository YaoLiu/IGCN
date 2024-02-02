import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
import igcn as gcn3d

class GCN3D(nn.Module):
    def __init__(self, class_num, support_num, neighbor_num):
        super().__init__()

        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(4, 32, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(32, 32, support_num= support_num)
        self.conv_2 = gcn3d.Conv_layer(64, 32, support_num= support_num)

        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)

        self.conv_3 = gcn3d.Conv_layer(96, 32, support_num= support_num)
        self.conv_4 = gcn3d.Conv_layer(128, 32, support_num= support_num)
        self.conv_5 = gcn3d.Conv_layer(160, 32, support_num= support_num)
        
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)

        self.conv_6 = gcn3d.Conv_layer(192, 32, support_num= support_num)
        self.conv_7 = gcn3d.Conv_layer(224, 32, support_num= support_num)

        dim_fuse = sum([32, 64, 96, 128, 160, 192, 224, 256, 256, 16])
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 1024, 1),
            nn.ReLU(inplace= False),
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace= False),
            nn.Conv1d(512, class_num, 1),
        )

    def forward(self, 
                vertices: "tensor (bs, vetice_num, 3)", 
                onehot: "tensor (bs, cat_num)"):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()
        # neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        neighbor_index = gcn3d.query_ball_point(0.25, self.neighbor_num, vertices, vertices)

        fm_0 = self.conv_0(neighbor_index, vertices, vertices)
        fm_0 = F.relu(fm_0, inplace= False)

        # fm_1 = F.relu(self.conv_1(neighbor_index, vertices, fm_0), inplace= True)
        fm_1 = self.conv_1(neighbor_index, vertices, fm_0)
        fm_1 = F.relu(fm_1, inplace= False)
        fm_1 = torch.cat((fm_0, fm_1), 2)

        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace= False) 
        fm_2 = torch.cat((fm_1, fm_2), 2)

        # v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_2, 0.25)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = gcn3d.query_ball_point(0.39, self.neighbor_num, v_pool_1, v_pool_1)

        # fm_2 = F.relu(self.conv_2(neighbor_index, v_pool_1, fm_pool_1), inplace= True)
        # fm_3 = F.relu(self.conv_3(neighbor_index, v_pool_1, fm_2), inplace= True)


        fm_3 = self.conv_3(neighbor_index, v_pool_1, fm_pool_1)
        fm_3 = F.relu(fm_3, inplace= False) 
        fm_3 = torch.cat((fm_pool_1, fm_3), 2)

        fm_4 = self.conv_4(neighbor_index, v_pool_1, fm_3)
        fm_4 = F.relu(fm_4, inplace= False) 
        fm_4 = torch.cat((fm_3, fm_4), 2)

        fm_5 = self.conv_5(neighbor_index, v_pool_1, fm_4)
        fm_5 = F.relu(fm_5, inplace= False) 
        fm_5 = torch.cat((fm_4, fm_5), 2)

        # v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_5, 0.39)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num)
        neighbor_index = gcn3d.query_ball_point(0.63, self.neighbor_num, v_pool_2, v_pool_2)

        # fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)

        fm_6 = self.conv_6(neighbor_index, v_pool_2, fm_pool_2)
        fm_6 = F.relu(fm_6, inplace= False) 
        fm_6 = torch.cat((fm_pool_2, fm_6), 2)

        fm_7 = self.conv_7(neighbor_index, v_pool_2, fm_6)
        fm_7 = F.relu(fm_7, inplace= False) 
        fm_7 = torch.cat((fm_6, fm_7), 2)

        # f_global = fm_4.max(1)[0] #(bs, f)
        f_global = fm_7.max(1)[0]

        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)


        fm_3 = gcn3d.indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d.indexing_neighbor(fm_4, nearest_pool_1).squeeze(2)
        fm_5 = gcn3d.indexing_neighbor(fm_5, nearest_pool_1).squeeze(2)

        fm_6 = gcn3d.indexing_neighbor(fm_6, nearest_pool_2).squeeze(2)
        fm_7 = gcn3d.indexing_neighbor(fm_7, nearest_pool_2).squeeze(2)

        f_global = f_global.unsqueeze(1).repeat(1, vertice_num, 1)
        onehot = onehot.unsqueeze(1).repeat(1, vertice_num, 1) #(bs, vertice_num, cat_one_hot)
        fm_fuse = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, fm_5, fm_6, fm_7, f_global, onehot], dim= 2)

        conv1d_input = fm_fuse.permute(0, 2, 1) #(bs, fuse_ch, vertice_num)
        conv1d_out = self.conv1d_block(conv1d_input) 
        pred = conv1d_out.permute(0, 2, 1) #(bs, vertice_num, ch)
        return pred

def test():
    from dataset_shapenet import test_model
    dataset = "../../shapenetcore_partanno_segmentation_benchmark_v0"
    model = GCN3D(class_num= 50, support_num= 1, neighbor_num= 50)
    test_model(model, dataset, cuda= "0", bs= 2, point_num= 2048)

if __name__ == "__main__":
    test()