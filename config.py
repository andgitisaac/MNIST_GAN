import argparse

def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aux", default=False, dest="AUX_CLASSIFIER", action="store_true")
    parser.add_argument("--cuda", default=False, dest="USE_CUDA", action="store_true")
    parser.add_argument("--workers", default=4, type=int, dest="NUM_WORKERS")

    parser.add_argument("--seed", default=9487, type=int, dest="SEED")
    parser.add_argument("--shuffle", default=True, type=bool, dest="SHUFFLE")

    parser.add_argument("--classes", default=10, type=int, dest="NUM_CLASSES")
    parser.add_argument("--zdim", default=100, type=int, dest="ZDIM")

    parser.add_argument("--epoch", default=10, type=int, dest="EPOCHS")
    parser.add_argument("--batch-size", default=100, type=int, dest="BATCH_SIZE")
    parser.add_argument("--lr", default=0.001, type=float, dest="LR")

    parser.add_argument("--log-step", default=20, type=int, dest="LOG_STEP")

    args = parser.parse_args()
    return args


def parse_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aux", default=False, dest="AUX_CLASSIFIER", action="store_true")
    parser.add_argument("--cuda", default=False, dest="USE_CUDA", action="store_true")

    parser.add_argument("--seed", default=9487, type=int, dest="SEED")

    parser.add_argument("--classes", default=10, type=int, dest="NUM_CLASSES")
    parser.add_argument("--zdim", default=100, type=int, dest="ZDIM")

    parser.add_argument("--epoch", default=9, type=int, dest="EPOCH")
    parser.add_argument("--batch-size", default=100, type=int, dest="BATCH_SIZE")

    args = parser.parse_args()
    return args