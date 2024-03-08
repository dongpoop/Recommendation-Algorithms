import matplotlib.pyplot as plt

import test
import test_SVD


def test_normal():
    """ 展示传统的推荐结果 """
    score_list, quality_list, number = test.test_accuracy(5, 0.8, 0.7, 'ml100k', 10000)
    num = range(1, number + 1)
    plt.plot(num, score_list)
    plt.xlabel("experiment number")
    plt.ylabel("precision")
    plt.title("precision")
    plt.show()
    plt.plot(num, quality_list)
    plt.xlabel("experiment number")
    plt.ylabel("quality")
    plt.title("quality")
    plt.show()


def test_svd():
    """ 展示基于SVD的算法表现 """
    score_list, quality_list, number = test_SVD.test_accuracy(5, 0.8, 0.7, 'ml100k', 10000)
    num = range(1, number + 1)
    plt.plot(num, score_list)
    plt.xlabel("experiment number")
    plt.ylabel("precision(svd)")
    plt.title("precision SVD")
    plt.show()
    plt.plot(num, quality_list)
    plt.xlabel("experiment number")
    plt.ylabel("quality(svd)")
    plt.title("quality SVD")
    plt.show()
