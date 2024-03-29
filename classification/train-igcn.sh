python3 main.py \
-mode train \
-rate 1.0 \
-support 4 \
-neighbor 32 \
-epoch 500 \
-lr 0.001 \
-dataset /home/yaoliu/scratch/data/modelnet/ModelNet40SampleNorm/ \
-bs 32 \
-interval 10 \
-cuda 7 \
-record ./experiment7/igcn-mm2/record.log \
-save ./experiment7/igcn-mm2/model.pkl \