import os
import sys
sys.path.append(os.path.abspath('../src'))

from tensorshare import TensorShare, run_server, RLWorker, RLTrainer
from threading import Timer
import logging
from time import sleep
from twisted.python import log
import torch


def test_tensorshare():
    logging.basicConfig(level=logging.DEBUG)
    # log.startLogging(sys.stdout)

    run_server(8000, start_reactor=False)

    ts = TensorShare('localhost', 8000)

    def _test():
        ts.append('bin1', [torch.randn(10, 2)], lambda x: print(x))
        ts.append('bin1', [torch.randn(10, 2)], lambda x: print(x))
        ts.list(lambda x: print(x))
        ts.peek('bin1', lambda x: print(x))
        ts.get('bin1', lambda x: print(x))
        ts.list(lambda x: print(x))
        ts.listen('bin2', lambda x: print("update on:", x))
        ts.append('bin2', [torch.randn(10, 2)], lambda x: print(x))
        ts.list(lambda x: print(x))
        ts.delete('bin2', lambda x: print(x))
        ts.list(lambda x: print(x))
        ts.get('bin2', lambda x: print(x))  # This should fail
        ts.ignore('bin2', lambda x: print(x))
        ts.put('bin2', ['some_data', torch.randn(10, 2)], lambda x: print(x))
        ts.get('bin2', lambda x: print(x))
        sleep(2)
        ts.stop()

    Timer(0.1, _test).start()
    ts.start()


def test_rl_wrapper(is_trainer: bool):
    # logging.basicConfig(level=logging.DEBUG)
    if is_trainer:
        trainer = RLTrainer('localhost', 8000, host_server=True, filter_version=True)
        try:

            def _test():
                # Publish initial parameters
                version = trainer.publish_parameters(torch.randn(10))
                print("Published parameter version: {}".format(version))
                # Fetch new data until we have aggregated enough for a new batch
                while True:
                    data, versions = trainer.get_data(return_versions=True)
                    if len(data) != 0:
                        break
                assert len(data) == len(versions)
                print("Received data: {} with versions {}".format(data, versions))
                version = trainer.publish_parameters(torch.randn(10))
                print("Published parameter version: {}".format(version))
                sleep(1)
                print('trainer.get_data:', trainer.get_data(return_versions=True))  # Version filtering
                sleep(3)
                trainer.stop()

            Timer(0.1, _test).start()
            trainer.start()
        finally:
            trainer.stop()
    else:
        worker = RLWorker('localhost', 8000, False)
        try:
            def _test():
                params = worker.get_parameters()
                if params is None:
                    print("awaiting params")
                    params = worker.await_new_parameters()
                print("Got initial params: {}, version: {}, {}".format(params, worker._last_retrieved_version,
                                                                       worker._latest_version))
                # Generate and publish new rollouts / self-play data
                sleep(0.5)
                worker.add_data([torch.randn(5, 2)])
                sleep(0.5)
                worker.add_data([torch.randn(5, 2)])
                # Await new parameters
                params = worker.await_new_parameters()
                print("Got new params: {}, version: {}, {}".format(params, worker._last_retrieved_version,
                                                                   worker._latest_version))
                sleep(3)
                worker.stop()

            Timer(0.1, _test).start()
            worker.start()
        finally:
            worker.stop()


if __name__ == '__main__':
    # test_tensorshare()
    test_rl_wrapper(bool(int(sys.argv[1])))
