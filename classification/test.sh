# python3 main.py \
# -mode test \
# -cuda 1 \
# -bs 8 \
# -dataset /home/yaoliu/scratch/data/modelnet/ModelNet40SampleNorm512/ \
# -support 1 \
# -neighbor 20 \
# -load ./experiment4/3dgcn-2/model-3dgcn.pkl \
# -model 3dgcn 

#-normal \
#-random \
#-rotate 180 \
#-scale 2 \
#-shift 10.0 \


python3 main.py \
-mode test \
-cuda 2 \
-bs 32 \
-dataset /home/yaoliu/scratch/data/modelnet/ModelNet40SampleNorm/ \
-support 4 \
-neighbor 32 \
-load ./experiment6/igcn0002/model.pkl \
# -model 3dgcn 