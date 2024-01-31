################### PRE TRAIN >>>>>>>>>>>>>>>>>>>>>>>>
# Command to run on Interactive test for DALI
mpirun --bind-to socket -np 1 python pretrain_dali.py $SGE_LOCALDIR/cifar100/train --dali --model deit_tiny_patch16_224 --experiment test_interactive  --input-size 3 224 224  --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --color-jitter 0.4 --hflip 0.5 --vflip 0.5 --scale 0.08 1.0 --ratio 0.75 1.3333  --opt adamw --lr 1.0e-3 --weight-decay 0.05 --deit-scale --sched cosine_iter --min-lr 1.0e-5 --warmup-lr 1e-06 --warmup-epochs 5 --warmup-iter 5000 --cooldown-epochs 0 --aa rand-m9-mstd0.5-inc1  --interpolation bicubic --reprob 0.5 --remode pixel --drop-path 0.1 --num-classes 1000  --eval-metric loss --log-interval 1 --no-prefetcher --amp --batch-size 32 -j 19 --epochs 10

# Command to run Pytorch NO AUGMENT
mpirun --bind-to socket -np 1 python pretrain.py ssd/FractalDB-1000-EGL-GLFW --model deit_tiny_patch16_224 --experiment test_interactive  --input-size 3 224 224  --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --opt adamw --lr 1.0e-3 --weight-decay 0.05 --deit-scale 512.0 --deit-scale --sched cosine_iter --min-lr 1.0e-5 --warmup-lr 1e-06 --warmup-epochs 5 --warmup-iter 5000 --cooldown-epochs 0 --num-classes 1000 --eval-metric loss --log-interval 50 --no-aug --interpolation nearest --train-interpolation nearest --no-prefetcher --amp --batch-size 64 -j 19 --epochs 1 --pin-mem

# Command to run Pytorch ALL
mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST -npernode 4 -np 16 python pretrain.py ssd/FractalDB-1000-EGL-GLFW --model deit_tiny_patch16_224 --experiment test_interactive  --input-size 3 224 224  --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --std 0.5 0.5 0.5 --color-jitter 0.4 --hflip 0.5 --vflip 0.5 --scale 0.08 1.0 --ratio 0.75 1.3333 --opt adamw --lr 1.0e-3 --weight-decay 0.05 --deit-scale 512.0 --sched cosine_iter --min-lr 1.0e-5 --warmup-lr 1e-06 --warmup-epochs 5 --warmup-iter 5000 --cooldown-epochs 0  --aa rand-m9-mstd0.5-inc1  --interpolation bicubic --num-classes 1000 --eval-metric loss --log-interval 50  --mixup 0.8 --cutmix 1.0 --drop-path 0.1  --reprob 0.5 --remode pixel --no-prefetcher --amp --pin-mem --batch-size 32 -j 19 --epochs 5

################### FINE TUNE ######################

# Command to fine-tune CIFAR100 PyTorch ALL
mpirun --bind-to socket -np 8 python finetune.py $SGE_LOCALDIR/cifar100  --model deit_tiny_patch16_224 --experiment test_interactive --input-size 3 224 224 --num-classes 100 --batch-size 96 --opt sgd --lr 0.01 --weight-decay 0.0001 --deit-scale 512.0  --sched cosine --lr-cycle-mul 1.0 --min-lr 1e-05 --decay-rate 0.1 --warmup-lr 1e-06 --warmup-epochs 10  --lr-cycle-limit 1 --cooldown-epochs 0 --scale 0.08 1.0 --ratio 0.75 1.3333 --hflip 0.5 --color-jitter 0.4 --interpolation bicubic --train-interpolation bicubic --mean 0.485 0.456 0.406  --std 0.229 0.224 0.225 --reprob 0.25 --recount 1 --remode pixel --aa rand-m9-mstd0.5-inc1  --mixup 0.8 --cutmix 1.0 --mixup-prob 1.0 --mixup-switch-prob 0.5 --mixup-mode batch --smoothing 0.1 --drop-path 0.1 -j 19 --no-prefetcher --amp --pin-mem --epochs 1 --crop-pct 1.0

# Command to fine-tune - DALI and PyTorch
mpirun --bind-to socket -np 4 python finetune.py $SGE_LOCALDIR/cifar100  --model deit_tiny_patch16_224 --experiment test_interactive --input-size 3 224 224 --num-classes 100 --batch-size 192 --opt sgd --lr 0.01 --weight-decay 0.0001 --deit-scale 512.0 --sched cosine --lr-cycle-mul 1.0 --min-lr 1e-05 --decay-rate 0.1 --warmup-lr 1e-06 --warmup-epochs 10  --lr-cycle-limit 1 --cooldown-epochs 0 --scale 0.08 1.0 --ratio 0.75 1.3333 --hflip 0.5 --color-jitter 0.4 --interpolation nearest --train-interpolation nearest --mean 0.485 0.456 0.406  --std 0.229 0.224 0.225 --reprob 0.25 --recount 1 --remode pixel --aa rand-m9-mstd0.5-inc1  --mixup 0.8 --cutmix 1.0 --mixup-prob 1.0 --mixup-switch-prob 0.5 --mixup-mode batch --smoothing 0.1 --drop-path 0.1 -j 19 --no-prefetcher --amp --pin-mem --epochs 1000 --crop-pct 0.875 --pretrained-path /home/acc12930pb/working/transformer/beforedali_timm_main_sora/checkpoint/tiny/fdb1k/pre_training/pretrain_deit_tiny_fdb1k_lr1.0e-3_epochs300_bs512_ssd_362x_GLFW3090/last.pth.tar

# To get best accuracy on Resnet18 PyTorch CIFAR100 -> 76.79 %
# Deit Scale is at 256
mpirun --bind-to socket -np 8 python finetune.py ssd/cifar100  --model resnet50 --experiment test_interactive_0 --input-size 3 224 224 --num-classes 100 --batch-size 128 --opt sgd --lr 0.4 --weight-decay 0.0001 --deit-scale --scale 0.08 1.0 --ratio 0.75 1.3333 --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 --color-jitter 0.0 --crop-pct 0.875 --interpolation bicubic --train-interpolation random -j 19 --no-prefetcher --amp --pin-mem --epochs 300

# To get best accuracy on Resnet18 PyTorch CIFAR100 -> 76.43 %
# Deit Scale is at 256
# 3interpolation for training and Triangular for eval
mpirun --bind-to socket -np 8 python finetune_Dali_All.py $SGE_LOCALDIR/cifar100 --dali  --model resnet18 --experiment test_interactive_0 --input-size 3 224 224 --num-classes 100 --batch-size 16 --opt sgd --lr 0.4 --weight-decay 0.0001 --deit-scale 512.0 --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 -j 19 --no-prefetcher --amp --pin-mem --epochs 300 --log-interval 100

# Fine-tune DAlI to CIFAR100 
mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST -np 4 python finetune_Dali_All.py $SGE_LOCALDIR/cifar100 --dali --model deit_tiny_patch16_224 --experiment test_interactive --input-size 3 224 224 --num-classes 100 --batch-size 96 --opt sgd --lr 0.01 --weight-decay 0.0001 --deit-scale 512.0 --sched cosine --lr-cycle-mul 1.0 --min-lr 1e-05 --decay-rate 0.1 --warmup-lr 1e-06 --warmup-epochs 10  --lr-cycle-limit 1 --cooldown-epochs 0 --mixup 0.8 --cutmix 1.0 --mixup-prob 1.0 --mixup-switch-prob 0.5 --mixup-mode batch --smoothing 0.1 --drop-path 0.1 -j 19 --no-prefetcher --amp --pin-mem

############## TEST TORCH-CUDA
python -c "import torch; print(torch.__version__); print(torch.__config__.show()); print(torch.__config__.parallel_info())"