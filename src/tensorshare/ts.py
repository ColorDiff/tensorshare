import logging
from time import sleep
from typing import Optional, Callable, List, Union, Tuple
from queue import Queue, Empty
from threading import Thread, Lock, Condition

from tensorshare.ftp import ClientCreator, InMemoryFTPClient, reactor, FileSender, BufferingProtocol,\
    NotifyListenerFactory, run_server
import io
import torch

_logger = logging.getLogger(__name__)


class TensorShare:
    """ Creates a tensorshare instance.

    Args:
        host (str): the host of the tensorshare server
        port (int): the port of the tensorshare server

    The instance will try to connect to the server with the given ip and port.
    Requests are stored in a queue and processed asynchronously.
    Request results are delivered via callbacks provided with the requests.
    Requests will be worked off after starting the tensorshare instance by calling the :meth:`start <tensorshare.ts.TensorShare.start>` method.
    :meth:`start <tensorshare.ts.TensorShare.start>` is blocking and must be called in the main thread like so:

    .. code-block:: python

        from tensorshare import TensorShare, run_server
        from threading import Thread

        def your_code_here():
            try:
                pass
            finally:
                ts.stop()

        # This starts the server in the same process.
        # Ideally this is run in a separate process.
        run_server(8000, start_reactor=False)
        ts = TensorShare('localhost', 8000)
        Thread(target=your_code_here, deamon=True).start()
        ts.start()

    All processed requests report the result to the callbacks provided with the requests.
    All callback will be called with a dictionary containing the key 'status', which is either 'OK' (see :attr:`STATUS_OK <tensorshare.ts.TensorShare.STATUS_OK>`)
    if the request has been processed successfully, or 'ERR' (see :attr:`STATUS_ERR <tensorshare.ts.TensorShare.STATUS_ERR>`)
    if the request has failed for some reason.
    If the request failed, the reason is provided in the 'msg' field.
    If any data is returned by the request (e.g. :meth:`get <tensorshare.ts.TensorShare.get>`, :meth:`peek <tensorshare.ts.TensorShare.peek>`, :meth:`list <tensorshare.ts.TensorShare.list>` methods), it will be returned in the 'data' field.

    .. code-block:: python

        from tensorshare import TensorShare, run_server
        from threading import Thread


        def example(ts):
            # Define callbacks
            def cb_put(resp):
                print("put response: Status =", resp['status'])
                if resp['status'] == ts.STATUS_ERR:
                    print("Got error:", resp['msg'])
                # Call list after put is finished
                ts.list(cb_list)

            def cb_list(resp):
                print("list response: Status =", resp['status'])
                if resp['status'] == ts.STATUS_ERR:
                    print("Got error:", resp['msg'])
                else:
                    print('list result:', resp['data']) # -> {'bin1': 1}
                # Shut down tensorshare, also stops run_server
                ts.stop()

            # Request to put data, calling cb_put with the result
            ts.put('bin1', list(range(5)), cb_put)

        run_server(8000, start_reactor=False)
        tmp = TensorShare('localhost', 8000)
        Thread(target=example, args(tmp, ), daemon=True).start()
        try:
            tmp.start()
        finally:
            tmp.stop()

    `tensorshare` supports any data-type that can be processed by
    `torch.load <https://pytorch.org/docs/stable/generated/torch.load.html#torch.load>`_ and
    `torch.save <https://pytorch.org/docs/stable/generated/torch.save.html#torch.save>`_.
    """

    STATUS_OK = 'OK'
    STATUS_ERR = 'ERR'

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._q = Queue()
        self._running = False
        self._clt: Optional[InMemoryFTPClient] = None
        self._listener: Optional[NotifyListenerFactory] = None
        self._th = Thread(target=self._run, daemon=True)
        self._wait = False
        self._listener_cbs = {}

    def append(self, bin: str, obj: object, callback: Callable[[dict], None] = None):
        """Append the given object to the objects in the specified bin.

        If the request is successful and the bin is empty, this method behaves identically to :meth:`put <tensorshare.ts.TensorShare.put>`.
        If there is exactly one item in the bin, both objects are put in a list.
        Otherwise, the given object is appended to the pre-existing list.
        In any successful case, the callback is called with the following message :code:`{'status': 'OK'}`

        If any error occurs, the callback will instead be called with this message :code:`{'status': 'ERR', 'msg': '<err_msg>'}`

        Args:
            bin (str): the bin to which the object will be appended
            obj (object): the object to append
            callback (Callable[[dict], None]) = None: A callback receiving the server's response.

        """
        if callback is None:
            callback = self._cb_pass
        self._q.put_nowait(('append', (bin, obj, callback)))

    def put(self, bin: str, obj: object, callback: Callable[[dict], None] = None):
        """Put the given object in the specified bin.

        If the request is successful and any data in the bin exists, it will be overwritten.
        The callback will be called with following message: :code:`{'status': 'OK'}`.

        If the request fails, the response will be formatted like :code:`{'status': 'ERR', 'msg': '<err_msg>'}`.

        Args:
            bin (str): the bin to which the object will be appended
            obj (object): the object to append
            callback (Callable[[dict], None]): A callback receiving the server's response.

        """
        if callback is None:
            callback = self._cb_pass
        self._q.put_nowait(('put', (bin, obj, callback)))

    def get(self, bin: str, callback: Callable[[dict], None]):
        """Get and remove all data in the specified bin.

        If the bin does exist, all data in the bin is given to the callback as either a list of objects
        (if any appending happened in the bin), or as the object itself in the 'data' field :code:`{'status': 'OK', 'data': <bin_content>}`
        All data in the bin will be removed after the request has been processed by the server.

        If the bin is empty or any other error occurs, the request fails and call the callback :code:`{'status': 'ERR', 'msg': '<err_msg>'}`

        Args:
            bin (str): the bin from which all objects are retrieved
            callback (Callable[[dict], None]): A callback receiving the server's response.

        """
        self._q.put_nowait(('get', (bin, callback, True)))

    def peek(self, bin, callback: Callable[[dict], None]):
        """Get and keep all data in the specified bin.

        If the bin does exist, all data in the bin is given to the callback as either a list of objects
        (if any appending happened in the bin), or as the object itself in the 'data' field :code:`{'status': 'OK', 'data': <bin_content>}`

        If the bin does not exist or is empty or any other error occured, the request fails, calling the callback with
        a message formatted like :code:`{'status': 'ERR', 'msg': '<err_msg>'}`.

        Args:
            bin (str): the bin from which all objects are retrieved
            callback (Callable[[dict], None]): A callback receiving the server's response.

        """
        self._q.put_nowait(('get', (bin, callback, False)))

    def list(self, callback: Callable[[dict], None]):
        """List all bins and the number of items within each.

        If successful, the listing is given as a dictionary, to the callback in the 'data' field:
         :code:`{'status': 'OK', 'data': {'<bin_name>': <bin_length>}}`

        .. code-block:: python

            ts.put('bin1', [1,2,3])
            ts.put('bin1', [2,3,4])  # Overwrites all contents of bin1
            ts.list(lambda x: print(x['data']) # yields {'bin1': 1}
            ts.append('bin1', [3,4,5])  # Appends data to the bin
            ts.list(lambda x: print(x['data']) # yields {'bin1': 2}
            ts.peek('bin1', lambda x: print(x['data'])
            ts.list(lambda x: print(x['data']) # yields {'bin1': 2}
            ts.get('bin1', lambda x: print(x['data'])
            ts.list(lambda x: print(x['data']) # yields {}

        If the request fails, the response will be formatted like:
         :code:`{'status': 'ERR', 'msg': '<err_msg>'}`


        Args:
            callback (Callable[[dict], None]): A callback receiving the server's response.

        """
        self._q.put_nowait(('list', (callback,)))

    def delete(self, bin: str, callback: Callable[[dict], None] = None):
        """Delete all data in the bin.

        If no data is in the bin or the bin does not exist, this is a no-op.

        Args:
            bin (str): which bin to delete
            callback (Callable[[dict], None]): A callback receiving the server's response.
        """
        if callback is None:
            callback = self._cb_pass
        self._q.put_nowait(('del', (bin, callback)))

    def listen(self, bin: str, callback: Callable[[dict], None]):
        """Start receiving write updates on the specified bin.

        Args:
            bin (str): the bin from which all objects are retrieved
            callback (Callable[[dict], None]): A callback receiving the server's response.

        Whenever a client modifies the specified bin, a message will be sent to the callback:
        :code:`{'status': 'OK', 'bin': '<bin_name>', 'n_items': len(<bin_name>)}`.

        If any error occurs during setup, the callback is called with status 'ERR':
        :code:`{'status': 'ERR', 'msg': '<err_msg>'}`

        This method only works if the current machine is visible to the server.
        """
        self._q.put_nowait(('listen', (bin, callback)))

    def ignore(self, bin: str, callback: Callable[[dict], None] = None):
        """Stop receiving write notifications on the given bin.

        Args:
            bin (str): the bin from which all objects are retrieved
            callback (Callable[[dict], None]): A callback receiving the server's response.
        """
        if callback is None:
            callback = self._cb_pass
        self._q.put_nowait(('ignore', (bin, callback)))

    def start(self):
        """Starts the client.

        This method is blocking and must be called from the `main thread`.
        """
        # Setup connection
        self._listener = NotifyListenerFactory(self._bin_updated)
        self._listener.listen(0)  # Listen on same port as the FTP server
        creator = ClientCreator(reactor, InMemoryFTPClient, 'anonymous', '')
        creator.connectTCP(self._host, self._port).addCallback(self._cb_set_clt).addErrback(self._cb_fail_connect)
        # Start thread
        self._running = True
        self._th.start()
        reactor.run()

    def stop(self):
        """Stops the client."""
        self._running = False
        if self._clt is not None:
            self._listener.quit()
            d = self._clt.quit()
            d.addBoth(lambda x: reactor.stop())

    def _bin_updated(self, info):
        bin, num = info
        if bin not in self._listener_cbs.keys():
            _logger.warning('Got update for unregistered bin: {}'.format(bin))
        else:
            self._listener_cbs[bin]({'status': self.STATUS_OK, 'bin': bin, 'n_items': num})

    def _run(self):
        while self._running:
            # Wait until the connection to the server has been established
            if self._clt is None:
                sleep(0.1)
                continue
            # Get task
            try:
                cmd, args = self._q.get(timeout=0.1)
            except Empty:
                continue
            _logger.debug('Cmd: {}, Current queue length: {}'.format(cmd, self._q.qsize()))

            # Work off the task
            if cmd == 'append':
                bin, obj, callback = args

                def tmp():
                    d, _ = self._clt.appendFile(bin)
                    d.addCallback(self._cb_send_file, self._to_bytes(obj), callback)
                    d.addErrback(self._cb_fail, 'Failed to append data: {}', callback)
            elif cmd == 'put':
                bin, obj, callback = args

                def tmp():
                    d, _ = self._clt.storeFile(bin)
                    d.addCallback(self._cb_send_file, self._to_bytes(obj), callback)
                    d.addErrback(self._cb_fail, 'Failed to put data: {}', callback)
            elif cmd == 'get':
                bin, callback, rm = args

                def tmp():
                    protocol = BufferingProtocol()
                    if rm:
                        d = self._clt.retrieveFile(bin, protocol)
                    else:
                        d = self._clt.peekFile(bin, protocol)
                    d.addCallback(self._cb_recv_file, protocol, callback)
                    d.addErrback(self._cb_fail, 'Failed to retrieve data: {}', callback)
            elif cmd == 'list':
                callback, = args

                def tmp():
                    protocol = BufferingProtocol()
                    d = self._clt.list(protocol)
                    d.addCallback(self._cb_recv_list, protocol, callback)
                    d.addErrback(self._cb_fail, 'Failed to list data: {}', callback)
            elif cmd == 'del':
                bin, callback = args

                def tmp():
                    d = self._clt.removeFile(bin)
                    d.addCallback(self._cb_success, callback)
                    d.addErrback(self._cb_fail, 'Failed to delete data: {}', callback)
            elif cmd == 'listen':
                bin, callback = args

                self._listener_cbs[bin] = callback

                def tmp():
                    d = self._clt.listen(bin + '|' + str(self._listener.get_port()))
                    d.addCallback(self._cb_success, callback, False)
                    d.addErrback(self._cb_fail, 'Failed to register as listener: {}', callback)
            elif cmd == 'ignore':
                bin, callback = args

                if bin in self._listener_cbs.keys():
                    del self._listener_cbs[bin]

                def tmp():
                    d = self._clt.ignore(bin)
                    d.addCallback(self._cb_success, callback)
                    d.addErrback(self._cb_fail, 'Failed to unregister as listener: {}', callback)
            else:
                _logger.warning('Received invalid cmd: {}'.format(cmd))
                continue

            self._wait = True

            # Invoke function from reactor thread
            reactor.callFromThread(tmp)

            # wait until the request has been handled
            while self._wait and self._running:
                sleep(0.01)
        _logger.info("Runner exited.")

    @staticmethod
    def _to_bytes(obj: object) -> io.BytesIO:
        res = io.BytesIO()
        res.seek(0)
        torch.save(obj, res)
        res.seek(0)
        return res

    @staticmethod
    def _from_bytes(byts: io.BytesIO) -> object:
        byts.seek(0)
        res = torch.load(byts, map_location='cpu')
        return res

    def _cb_pass(self, x):
        pass

    def _cb_send_file(self, consumer, file_obj, callback: callable):
        def cb_finish(_):
            consumer.finish()
            self._wait = False
            callback({'status': self.STATUS_OK})

        FileSender().beginFileTransfer(file_obj, consumer).addCallback(cb_finish) \
            .addErrback(self._cb_fail, 'Failed to send file: {}', callback)
        return consumer, file_obj

    def _cb_recv_file(self, x, protocol: BufferingProtocol, callback: callable):
        self._wait = False
        try:
            res = {'status': self.STATUS_OK,
                   'data': self._from_bytes(protocol.buffer)}
        except Exception as e:
            self._cb_fail(e, 'Failed to decode response: {}', callback)
            return x
        callback(res)
        return x

    def _cb_recv_list(self, x, protocol: BufferingProtocol, callback: callable):
        self._wait = False
        try:
            res = {'status': self.STATUS_OK,
                   'data': eval(protocol.buffer.getvalue().decode('ascii'))}
        except Exception as e:
            self._cb_fail(e, 'Failed to decode response: {}', callback)
            return x
        callback(res)
        return x

    def _cb_success(self, x, callback: callable, call_callback=True):
        self._wait = False
        if call_callback:
            callback({'status': self.STATUS_OK})
        return x

    def _cb_fail(self, err, msg: str, callback: callable):
        self._wait = False
        msg = msg.format(str(err))
        _logger.debug(msg)
        callback({'status': self.STATUS_ERR, 'msg': msg})
        return

    def _cb_set_clt(self, x):
        self._clt = x
        return

    def _cb_fail_connect(self, x):
        _logger.critical('Could not connect to the tensorshare server: {}.'.format(str(x)))
        self.stop()
        return


class RLWorker:
    """Creates a :class:`RLWorker <tensorshare.ts.RLWorker>` instance that provides commonly used functionality in DDRL for rollout or self-play workers.

    The :class:`RLWorker <tensorshare.ts.RLWorker>` supports two main functionalities:

    1. retrieving latest network parameters (see :meth:`get_parameters <tensorshare.ts.RLWorker.get_parameters>`, and :meth:`await_new_parameters <tensorshare.ts.RLWorker.await_new_parameters>`)
    2. appending self-play or rollout data to the servers storage (see :meth:`add_data <tensorshare.ts.RLWorker.add_data>`)

    A local copy of the current network parameters is kept and returned to any thread requesting them without delay via :meth:`get_parameters <tensorshare.ts.RLWorker.get_parameters>`.
    Once a write update on these occurs, the :class:`RLWorker <tensorshare.ts.RLWorker>` will update the local copy immediately.
    Additionally, the parameter version is tracked and any data added to the server includes this version, allowing
    the :class:`RLTrainer <tensorshare.ts.RLTrainer>` to filter for on-policy samples.

    Publishing of data is non-blocking for callers of :meth:`add_data <tensorshare.ts.RLWorker.add_data>`.

    Lastly, waiting for parameter updates is supported :meth:`await_new_parameters <tensorshare.ts.RLWorker.await_new_parameters>`.
    This may be useful if the :class:`RLTrainer <tensorshare.ts.RLTrainer>` has not yet published any parameters yet or
    if the worker knows how many samples to produce util a new version is expected.

    Args:
        host (str): The host machine of the tensorshare server
        port (int): The port on which the tensorshare server listens
        host_server (bool): Whether or not to host the server in this process. This is included for testing purposes and should not be used in real training scenarios. In real training scenarios, start the server using :func:`start_server <tensorshare.ftp.start_server>` in a seperate process.
    """

    def __init__(self, host: str, port: int, host_server: bool = False):
        if host_server:
            run_server(port, start_reactor=False)
        self.ts = TensorShare(host, port)
        self.ts.peek('parameters', self._cb_update_parameters)
        self.ts.listen('parameters', self._cb_parameters_updated)

        self._last_retrieved_version = -1
        self._latest_params = None
        self._latest_version = -1
        self._lock = Lock()
        self._cond = Condition()

    def start(self):
        """Starts connecting to the tensorshare server.

        This methods is blocking and must be called from the main thread.
        Make sure to have the server running before calling this method and to also call
        :meth:`close <tensorshare.ts.RLWorker.close>` when exiting (e.g. using the try-finally pattern).
        """
        # Blocking
        self.ts.start()

    def stop(self):
        """Ends the connection to the tensorshare server, freeing all resources in the process.

        This method should always be called after calling :meth:`start <tensorshare.ts.RLWorker.start>` to make sure
        the process can exit smoothly.
        """
        # This might still not finish if the server is hosted in this instance (until all conns are closed).
        self.ts.stop()

    def add_data(self, data: object):
        """Adds data to the tensorshare server, from which the :class:`RLTrainer <tensorshare.ts.RLTrainer>` can retrieve it.

        The data is send to the server in a separate process and is non-blocking.

        Args:
            data (object): The data to add to the tensorshare server.
        """
        with self._lock:
            self.ts.append('data', {'data': data, 'version': self._last_retrieved_version}, lambda x: None)

    def await_new_parameters(self, timeout: Optional[int] = None) -> Optional[object]:
        # If no new parameters have been send, return without waiting
        # Else we wait until an update comes.
        if self._last_retrieved_version == self._latest_version:
            with self._cond:
                self._cond.wait(timeout=timeout)
        return self.get_parameters()

    def get_parameters(self) -> Optional[object]:
        with self._lock:
            self._last_retrieved_version = self._latest_version
            return self._latest_params

    def _cb_parameters_updated(self, msg):
        if msg['status'] == 'OK' and len(msg) == 1:  # Initial connection success confirmation message
            return
        if msg['status'] != 'OK' or msg['bin'] != 'parameters':
            print("Got error on parameter update:", msg)
            return
        self.ts.peek('parameters', self._cb_update_parameters)

    def _cb_update_parameters(self, msg):
        if msg['status'] != 'OK':
            print("Got error while retrieving parameters", msg)
            return
        with self._lock:
            self._latest_params = msg['data']['params']
            self._latest_version = msg['data']['version']
        with self._cond:
            self._cond.notifyAll()


class RLTrainer:

    def __init__(self, host: str, port: int, host_server: bool, filter_version: bool = True):
        if host_server:
            run_server(port, start_reactor=False)
        self.ts = TensorShare(host, port)
        self._latest_version = -1
        self._data = []
        self._versions = []
        self._filter_version = filter_version
        self._lock = Lock()
        self._running = False
        self._th = Thread(target=self._run, daemon=True)

    def start(self):
        self._th.start()
        self.ts.start()

    def stop(self):
        self._running = False
        self.ts.stop()
        self._th.join()

    def get_data(self, return_versions: bool = False) -> Union[List[object], Tuple[List[object], List[int]]]:
        with self._lock:
            data = self._data
            versions = self._versions
            self._data = []
            self._versions = []
        if self._filter_version:
            data = [d for i, d in enumerate(data) if versions[i] >= self._latest_version]
            versions = [v for v in versions if v >= self._latest_version]
        if return_versions:
            return data, versions
        return data

    def publish_parameters(self, parameters: object) -> int:
        self._latest_version += 1
        self.ts.put('parameters', {'params': parameters, 'version': self._latest_version}, lambda x: None)
        return self._latest_version

    def _cb_add_data(self, msg):
        if msg['status'] != 'OK':
            # print("Error while retrieving data:", msg)
            return
        if 'data' not in msg.keys() or len(msg['data']) == 0:
            return
        with self._lock:
            # print("got data:", msg['data']['data'], msg['data']['version'])
            if isinstance(msg['data'], list):
                for item in msg['data']:
                    self._data.append(item['data'])
                    self._versions.append(item['version'])
            else:
                self._data.append(msg['data']['data'])
                self._versions.append(msg['data']['version'])

    def _run(self):
        self._running = True
        while self._running:
            self.ts.get('data', self._cb_add_data)
            sleep(0.5)
