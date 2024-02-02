python3 main.py \
-mode test \
-support 4 \
-neighbor 50 \
-load ./experiment6/pointnet/model.pkl \
-cuda 2 \
-bs 8 \
-dataset /home/yaoliu/scratch/data/shapenet/shapenetcore_partanno_segmentation_benchmark_v0 \
-point 1024 \
-output out_imgs_pointnet/ \
-model pointnet 
#-random \
#-rotate 0 \
# -axis 1 \
#-scale 0.0 \
#-shift 0.0 \

