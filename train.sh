# Vanilla GAN
python3 train.py --GAN --seed 6666 --batch-size 100 --lr 0.001 --epoch 20
# python3 train.py --GAN --AUG --seed 6666 --batch-size 100 --lr 0.001 --epoch 20 # Data Augmentation
# tensorboard --logdir=logs/GAN

# CGAN
python3 train.py --CGAN --aux --seed 6666 --batch-size 100 --lr 0.001 --epoch 20
# python3 train.py --CGAN --AUG --aux --seed 6666 --batch-size 100 --lr 0.001 --epoch 20 # Data Augmentation
# tensorboard --logdir=logs/CGAN

# ACGAN
python3 train.py --ACGAN --aux --seed 6666 --batch-size 100 --lr 0.001 --epoch 20
# python3 train.py --ACGAN --AUG --aux --seed 6666 --batch-size 100 --lr 0.001 --epoch 20 # Data Augmentation
# tensorboard --logdir=logs/ACGAN