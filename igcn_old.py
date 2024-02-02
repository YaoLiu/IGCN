"""
@Author: Zhi-Hao Lin
@Contact: r08942062@ntu.edu.tw
@Time: 2020/03/06
@Document: Basic operation/blocks of 3D-GCN
"""
# lanni version 10 6->10

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_neighbor_index(vertices: "(bs, vertice_num, 3)",  neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    bs, v, _ = vertices.size()
    device = vertices.device
    inner = torch.bmm(vertices, vertices.transpose(1, 2)) #(bs, v, v)
    quadratic = torch.sum(vertices**2, dim= 2) #(bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(distance, k= neighbor_num + 1, dim= -1, largest= False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    for i in torch.arange(B, dtype=torch.long):
        a = torch.diag(dist[i])
        a_diag = torch.diag_embed(a)
        dist[i] = dist[i] - a_diag
    return dist

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1]) #[B, S, N]
    sqrdists = square_distance(new_xyz, xyz) # [B, S, N]
    group_idx[sqrdists > radius ** 2 ] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx

def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"):
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target, source.transpose(1, 2)) #(bs, v1, v2)
    s_norm_2 = torch.sum(source ** 2, dim= 2) #(bs, v2)
    t_norm_2 = torch.sum(target ** 2, dim= 2) #(bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k= 1, dim= -1, largest= False)[1]
    return nearest_index

def indexing_neighbor(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)" ):
    """
    Return: (bs, vertice_num, neighbor_num, dim)
    """
    bs, v, n = index.size()
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index]
    return tensor_indexed

def get_neighbor_direction_norm(vertices: "(bs, vertice_num, 3)", neighbor_index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighobr_num, 3)
    """
    neighbors = indexing_neighbor(vertices, neighbor_index) # (bs, v, n, 3)
    neighbor_direction = neighbors - vertices.unsqueeze(2)
    neighbor_direction_norm = F.normalize(neighbor_direction, dim= -1)
    return neighbor_direction_norm

class Conv_surface(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments: 
        self.in_channel = in_channel # 4
        self.out_channel = out_channel # 32
        self.support_num = support_num # fiexd 4

        # function
        self.relu = nn.ReLU(inplace= False)
        self.bn2 = nn.BatchNorm1d(out_channel)

        self.nn = nn.Linear(in_channel, out_channel)

        # parameters:
        self.directions = nn.Parameter(torch.FloatTensor(4, out_channel)) # 4: support_num
        self.initialize()
        

    def initialize(self):
        stdv = math.sqrt(2. / self.out_channel)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self, 
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index) # (bs, vertice_num, neighobr_num, 3)
        support_direction_norm = self.directions # (4, out_channel)

        neighbor_direction_norm_weight = (neighbor_direction_norm+1)/2 # (bs, vertice_num, neighobr_num, 3)

        support_direction_norm_weight = torch.cat((
            (support_direction_norm[1] - support_direction_norm[0]).view(1,-1),
            (support_direction_norm[2] - support_direction_norm[0]).view(1,-1),
            (support_direction_norm[3] - support_direction_norm[0]).view(1,-1)
            ),0)

        theta = neighbor_direction_norm_weight @ support_direction_norm_weight + support_direction_norm[0] # (bs, vertice_num, neighbor_num, out_channel)
        theta = self.relu(theta)

        feature_support = indexing_neighbor(feature_map, neighbor_index) # (bs, vertice_num, neighbor_num, 3)
        feature_distance = feature_support.permute(3, 0, 1, 2) # (in_channel, bs, vertice_num, neighbor_num)
        feature_distance = torch.sqrt(torch.sum(torch.pow(feature_distance, 2), 0)) # (bs, vertice_num, neighbor_num)
        feature_distance = feature_distance.contiguous().view(bs, vertice_num, neighbor_num, 1) # (bs, vertice_num, neighbor_num, 1)
        feature_support = torch.cat((feature_support,feature_distance),3) # (bs, vertice_num, neighbor_num, 4)

        feature_support = feature_support.contiguous().view(-1, self.in_channel) # (bs, in_channel, vertice_num*neighbor_num)
        feature_support = self.relu(self.bn2(self.nn(feature_support))) # (bs, out_channel, vertice_num*neighbor_num)
        feature_support = feature_support.contiguous().view(bs, vertice_num, neighbor_num, -1)


        activation_support = feature_support * theta # (bs, vertice_num, neighbor_num, out_channel)
        activation_support = torch.max(activation_support, dim= 2)[0] # (bs, vertice_num, out_channel)  

        feature_fuse = activation_support # (bs, vertice_num, out_channel) 

        return feature_fuse

class Conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments: 
        self.in_channel = in_channel # 32, 64 ...
        self.out_channel = out_channel # 64, 128
        self.support_num = support_num # fiexd 4

        # function
        self.relu = nn.ReLU(inplace= False)
        self.bn2 = nn.BatchNorm1d(out_channel)

        self.nn = nn.Linear(in_channel, out_channel)

        # parameters:
        self.directions = nn.Parameter(torch.FloatTensor(4, out_channel)) # 4: support_num
        self.initialize()

    def initialize(self):
        stdv = math.sqrt(2. / self.out_channel)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self, 
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index) # (bs, vertice_num, neighobr_num, 3)
        support_direction_norm = self.directions # (4, out_channel)
        
        neighbor_direction_norm_weight = (neighbor_direction_norm+1)/2 # (bs, vertice_num, neighobr_num, 3)
 
        support_direction_norm_weight = torch.cat((
            (support_direction_norm[1] - support_direction_norm[0]).view(1,-1),
            (support_direction_norm[2] - support_direction_norm[0]).view(1,-1),
            (support_direction_norm[3] - support_direction_norm[0]).view(1,-1)
            ),0)
        

        theta = neighbor_direction_norm_weight @ support_direction_norm_weight + support_direction_norm[0] # (bs, vertice_num, neighbor_num, out_channel)
        theta = self.relu(theta) # (bs, vertice_num, neighbor_num, out_channel)
      


        feature_support = indexing_neighbor(feature_map, neighbor_index) # (bs, vertice_num, neighbor_num, in_channel)
        feature_support = feature_support.contiguous().view(-1, self.in_channel) # (bs, in_channel, vertice_num*neighbor_num)
        feature_support = self.relu(self.bn2(self.nn(feature_support))) # (bs, out_channel, vertice_num*neighbor_num)
        feature_support = feature_support.contiguous().view(bs, vertice_num, neighbor_num, -1)
 

        
        activation_support = feature_support * theta # (bs, vertice_num, neighbor_num, out_channel)
        activation_support = torch.max(activation_support, dim= 2)[0] # (bs, vertice_num, out_channel)  

        feature_fuse = activation_support # (bs, vertice_num, out_channel) 


        return feature_fuse
    

class Pool_layer(nn.Module):
    def __init__(self, pooling_rate: int= 4, neighbor_num: int=  4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

    def forward(self, 
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)",
                radius:"int"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice_num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()
        neighbor_index = query_ball_point(radius, self.neighbor_num, vertices, vertices)
        neighbor_feature = indexing_neighbor(feature_map, neighbor_index) #(bs, vertice_num, neighbor_num, channel_num)

        pooled_feature = torch.max(neighbor_feature, dim= 2)[0] #(bs, vertice_num, channel_num)
        
        pool_num = int(vertice_num / self.pooling_rate)
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :] # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, sample_idx, :] #(bs, pool_num, channel_num)
        return vertices_pool, feature_map_pool

def test():
    import time
    bs = 8
    v = 1024
    dim = 3
    n = 20
    vertices = torch.randn(bs, v, dim)
    neighbor_index = get_neighbor_index(vertices, n)
    """
    torch.Size([8, 1024, 20])
    """

    s = 3
    conv_1 = Conv_surface(kernel_num= 32, support_num= s)
    conv_2 = Conv_layer(in_channel= 32, out_channel= 64, support_num= s)
    pool = Pool_layer(pooling_rate= 4, neighbor_num= 4)
    
    print("Input size: {}".format(vertices.size()))
    start = time.time()
    f1 = conv_1(neighbor_index, vertices)
    print("\n[1] Time: {}".format(time.time() - start))
    print("[1] Out shape: {}".format(f1.size()))
    start = time.time()
    f2 = conv_2(neighbor_index, vertices, f1)
    print("\n[2] Time: {}".format(time.time() - start))
    print("[2] Out shape: {}".format(f2.size()))
    start = time.time()
    v_pool, f_pool = pool(vertices, f2)
    print("\n[3] Time: {}".format(time.time() - start))
    print("[3] v shape: {}, f shape: {}".format(v_pool.size(), f_pool.size()))
    
    
    """
    bash-3.2$ python gcn3d.py
    Input size: torch.Size([8, 1024, 3])

    [1] Time: 0.1121070384979248
    [1] Out shape: torch.Size([8, 1024, 32])

    [2] Time: 0.40329599380493164
    [2] Out shape: torch.Size([8, 1024, 64])

    [3] Time: 0.13983416557312012
    [3] v shape: torch.Size([8, 256, 3]), f shape: torch.Size([8, 256, 64])
    """


if __name__ == "__main__":
    test()
