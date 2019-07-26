from keras.datasets import mnist
from keras.utils import np_utils


def get_mnist():
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
    print('x_train.shape:', x_train.shape)
    print('y_train.shape:', y_train.shape)

    # reshape函数，如果填入-1， 将会自动适应
    # 要进行归一化处理，不然你的loss算出来会变成nan
    x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1)
    x_train = x_train / 255
    print('x_train.shape:', x_train.shape)
    x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)
    x_test = x_test / 255

    # 使用keras中np_utils将test转换为one_hot向量
    y_train = np_utils.to_categorical(y_train)
    print('y_train.shape:', y_train.shape)
    y_test = np_utils.to_categorical(y_test)
    return ((x_train, y_train), (x_test, y_test))