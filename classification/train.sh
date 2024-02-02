python3 main.py \
-mode train \
-support 4 \
-rate 0.8 \
-neighbor 32 \
-cuda 7 \
-epoch 100 \
-bs 32 \
-dataset /home/yaoliu/scratch/data/modelnet/ModelNet40SampleNorm/ \
-record ./experiment6/pointnet30021/record-pointnet.log \
-save ./experiment6/pointnet30021/model-pointnet.pkl \
-model pointnet 
#-normal