# Vanilla GAN
python3 train.py --seed 9487 --batch-size 100 --lr 0.001 --epoch 10
# tensorboard --logdir=logs/GAN

# ACGAN
python3 train.py --aux --seed 9487 --batch-size 100 --lr 0.001 --epoch 10
# tensorboard --logdir=logs/ACGAN