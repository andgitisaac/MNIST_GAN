DATA_DIR="data"
MNIST_TRAIN_IMAGE="train-images-idx3-ubyte"
MNIST_TRAIN_LABEL="train-labels-idx1-ubyte"
MNIST_TEST_IMAGE="t10k-images-idx3-ubyte"
MNIST_TEST_LABEL="t10k-labels-idx1-ubyte.gz"

mkdir ${DATA_DIR}
cd ${DATA_DIR}
curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O "${MNIST_TRAIN_IMAGE}.gz"
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O "${MNIST_TRAIN_LABEL}.gz"
curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O "${MNIST_TEST_IMAGE}.gz"
curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O "${MNIST_TEST_LABEL}.gz"

gunzip ${MNIST_TRAIN_IMAGE}
gunzip ${MNIST_TRAIN_LABEL}
gunzip ${MNIST_TEST_IMAGE}
gunzip ${MNIST_TEST_LABEL}
cd ..