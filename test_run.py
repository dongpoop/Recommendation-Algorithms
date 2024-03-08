import multi_process_experiment
import multiprocessing as mp


def test_process():
    """ 测试主进程 """
    process = [
        mp.Process(target=multi_process_experiment.test_normal),
        mp.Process(target=multi_process_experiment.test_svd),
    ]
    # 开启两个测试进程
    [p.start() for p in process]
    # 等待两个进程依次结束
    [p.join() for p in process]


if __name__ == '__main__':
    test_process()
