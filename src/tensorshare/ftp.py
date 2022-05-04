import threading
from typing import Optional

from twisted.protocols.ftp import FTPFactory, FTPRealm, FTPAnonymousShell, AnonUserDeniedError, FTP, IFTPShell, \
    FTPClient, FileNotFoundError, FileConsumer, BadCmdSequenceError, FTPCmdError, FILE_NOT_FOUND, TXFR_COMPLETE_OK, \
    CNX_CLOSED_TXFR_ABORTED, DATA_CNX_ALREADY_OPEN_START_XFR, FILE_STATUS_OK_OPEN_DATA_CNX, REQ_FILE_ACTN_COMPLETED_OK
from twisted.cred.portal import Portal
from twisted.cred.checkers import AllowAnonymousAccess
from twisted.python import log, filepath
from twisted.internet.protocol import Protocol, ClientCreator, Factory, ReconnectingClientFactory
from twisted.internet import reactor, defer
from twisted.protocols.basic import FileSender, LineReceiver
import io
import torch


class _BytesReader:
    def __init__(self, byts: bytes):
        self.byts = io.BytesIO(byts)
        self._send = False

    def _close(self, passthrough):
        self._send = True
        return passthrough

    def send(self, consumer):
        assert not self._send, "Can only call IReadFile.send *once* per instance"
        self._send = True
        self.byts.seek(0)
        d = FileSender().beginFileTransfer(self.byts, consumer)
        d.addBoth(self._close)
        return d


class NonClosingFileConsumer(FileConsumer):

    def unregisterProducer(self):
        self.producer = None


class _BytesWriter:
    def __init__(self, factory, path, append: bool):
        self.factory = factory
        self.path = path
        self.byts = io.BytesIO()
        self.append = append
        self._receive = False

    @staticmethod
    def decode(byts: io.BytesIO):
        byts.seek(0)
        res = torch.load(byts, map_location='cpu')
        return res

    @staticmethod
    def encode(obj: object):
        byts = io.BytesIO()
        torch.save(obj, byts)
        return byts

    def receive(self):
        assert not self._receive, "Can only call IWriteFile.receive *once* per instance"
        self._receive = True
        # FileConsumer will close the file object
        return defer.succeed(NonClosingFileConsumer(self.byts))

    def close(self):
        log.msg("recv bytes:", len(self.byts.getvalue()))
        try:
            if self.append and self.path in self.factory.storage:
                if not self.factory.decoded[self.path]:
                    obj = self.decode(
                        io.BytesIO(self.factory.storage[self.path])
                    )
                else:
                    obj = self.factory.storage[self.path]
                # Make a list if it is not one already
                if not isinstance(obj, list):
                    obj = [obj]
                to_append = self.decode(self.byts)
                obj.append(to_append)
                self.factory.storage[self.path] = obj
                self.factory.n_items[self.path] += 1
                self.factory.decoded[self.path] = True
            else:
                self.factory.storage[self.path] = self.byts.getvalue()
                self.factory.n_items[self.path] = 1
                self.factory.decoded[self.path] = False
            for cnx, paths in self.factory.cnxs.values():    # Notify listeners
                if self.path in paths and cnx is not None:
                    reactor.callLater(0, cnx.sendLine, self.path + '|' + str(self.factory.n_items[self.path]))
        except Exception as e:
            defer.fail(e)
        return defer.succeed(None)


class InMemoryFTPShell(FTPAnonymousShell):

    def __init__(self):
        self.factory = None
        super().__init__(filepath.FilePath('.'))

    def openForReading(self, path: str, rm=True):
        if self.factory is None:
            return defer.fail(None)
        if path not in self.factory.storage.keys():
            return defer.fail(FileNotFoundError(path))
        # Get the requested value and delete the corresponding storage path
        byts = self.factory.storage[path]
        is_decoded = self.factory.decoded[path]
        if rm:
            del self.factory.storage[path]
            del self.factory.n_items[path]
            del self.factory.decoded[path]
            for cnx, paths in self.factory.cnxs.values():  # Notify listeners
                if path in paths and cnx is not None:
                    reactor.callLater(0, cnx.sendLine, path + '|' + str(0))
        # Send the reader
        if is_decoded:
            byts = _BytesWriter.encode(byts).getvalue()
        return defer.succeed(_BytesReader(byts))

    def openForWriting(self, path):
        if self.factory is None:
            return defer.fail(None)
        return defer.succeed(_BytesWriter(self.factory, path, append=False))

    def openForAppending(self, path):
        if self.factory is None:
            return defer.fail(None)
        return defer.succeed(_BytesWriter(self.factory, path, append=True))

    def removeFile(self, path):
        if self.factory is None:
            return defer.fail(None)
        if path in self.factory.storage.keys():
            del self.factory.storage[path]
            del self.factory.n_items[path]
            del self.factory.decoded[path]
            for cnx, paths in self.factory.cnxs.values():    # Notify listeners
                if path in paths and cnx is not None:
                    reactor.callLater(0, cnx.sendLine, path + '|0')
        return defer.succeed(None)

    def access(self, path):
        return defer.fail(AnonUserDeniedError())

    def stat(self, path, keys=()):
        return defer.fail(AnonUserDeniedError())

    def list(self, path, keys=()):
        if self.factory is None:
            return defer.fail(None)
        # Get the requested value and delete the corresponding storage path
        res = {}
        for k in self.factory.storage.keys():
            res[k] = self.factory.n_items[k]
        byts = str(res).encode('ascii')
        # Send the reader
        return defer.succeed(_BytesReader(byts))


class InMemoryFTP(FTP):

    def ftp_PASS(self, password):
        d = super().ftp_PASS(password)

        def cb_set_factory(result):
            self.shell.factory = self.factory
            return result

        d.addCallback(cb_set_factory)
        return d

    def ftp_STOR(self, path, append=False):
        if self.dtpInstance is None:
            raise BadCmdSequenceError("PORT or PASV required before STOR")

        # XXX For now, just disable the timeout.  Later we'll want to
        # leave it active and have the DTP connection reset it
        # periodically.
        self.setTimeout(None)

        # Put it back later
        def enableTimeout(result):
            self.setTimeout(self.factory.timeOut)
            return result

        def cbOpened(file):
            """
            File was open for reading. Launch the data transfer channel via
            the file consumer.
            """
            dd = file.receive()
            dd.addCallback(cbConsumer)
            dd.addCallback(lambda ignored: file.close())
            dd.addCallbacks(cbSent, ebSent)
            return dd

        def ebOpened(err):
            """
            Called when failed to open the file for reading.

            For known errors, return the FTP error code.
            For all other, return a file not found error.
            """
            log.err(err, "Unexpected error received while opening file:")
            return FILE_NOT_FOUND, path

        def cbConsumer(cons):
            """
            Called after the file was opended for reading.

            Prepare the data transfer channel and send the response
            to the command channel.
            """

            dd = self.dtpInstance.registerConsumer(cons)

            # Tell them what to doooo
            if self.dtpInstance.isConnected:
                self.reply(DATA_CNX_ALREADY_OPEN_START_XFR)
            else:
                self.reply(FILE_STATUS_OK_OPEN_DATA_CNX)

            return dd

        def cbSent(result):
            """
            Called from data transport when transfer is done.
            """
            return TXFR_COMPLETE_OK,

        def ebSent(err):
            """
            Called from data transport when there are errors during the
            transfer.
            """
            log.err(err, "Unexpected error received during transfer:")
            if err.check(FTPCmdError):
                return err
            return CNX_CLOSED_TXFR_ABORTED,

        if append:
            d = self.shell.openForAppending(path)
        else:
            d = self.shell.openForWriting(path)

        d.addCallbacks(cbOpened, ebOpened)
        d.addBoth(enableTimeout)

        # Pass back Deferred that fires when the transfer is done
        return d

    def ftp_RETR(self, path, mode=''):
        if self.dtpInstance is None:
            raise BadCmdSequenceError("PORT or PASV required before RETR")

        # XXX For now, just disable the timeout.  Later we'll want to
        # leave it active and have the DTP connection reset it
        # periodically.
        self.setTimeout(None)

        # Put it back later
        def enableTimeout(result):
            self.setTimeout(self.factory.timeOut)
            return result

        # And away she goes
        cons = self.dtpInstance

        def cbSent(result):
            return TXFR_COMPLETE_OK,

        def ebSent(err):
            log.msg("Unexpected error attempting to transmit file to client:")
            log.err(err)
            if err.check(FTPCmdError):
                return err
            return CNX_CLOSED_TXFR_ABORTED,

        def cbOpened(file):
            # Tell them what to doooo
            if self.dtpInstance.isConnected:
                self.reply(DATA_CNX_ALREADY_OPEN_START_XFR)
            else:
                self.reply(FILE_STATUS_OK_OPEN_DATA_CNX)

            dd = file.send(cons)
            dd.addCallbacks(cbSent, ebSent)
            return dd

        def ebOpened(err):
            return FILE_NOT_FOUND, path

        if mode == 'list':
            d = self.shell.list(path)
        elif mode == 'peek':
            d = self.shell.openForReading(path, rm=False)
        else:
            d = self.shell.openForReading(path)
        d.addCallbacks(cbOpened, ebOpened)
        d.addBoth(enableTimeout)

        # Pass back Deferred that fires when the transfer is done
        return d

    def ftp_DELE(self, path):
        return self.shell.removeFile(path).addCallback(lambda ign: (REQ_FILE_ACTN_COMPLETED_OK,))

    def ftp_LIST(self, path=''):
        return self.ftp_RETR(path, mode='list')

    def ftp_APPD(self, path):
        return self.ftp_STOR(path, append=True)

    def ftp_PEEK(self, path):
        return self.ftp_RETR(path, mode='peek')

    def ftp_LSTN(self, path):
        path, port = path.split('|')
        port = int(port)
        if self.addr in self.factory.cnxs.keys():
            self.factory.cnxs[self.addr][1].append(path)
            return defer.succeed(None).addCallback(lambda x: (REQ_FILE_ACTN_COMPLETED_OK,))
        else:
            self.addr = self.addr[0], port
            self.factory.cnxs[self.addr] = (None, [path])
            self.factory.connect_to_listener(self.addr[0], port)
            return defer.succeed(None).addCallback(lambda x: (REQ_FILE_ACTN_COMPLETED_OK,))
        # return defer.fail(None)

    def ftp_NLSN(self, path):
        if self.addr in self.factory.cnxs.keys():
            cnx, paths = self.factory.cnxs[self.addr]
            self.factory.cnxs[self.addr] = cnx, [p for p in paths if p != path]
            return defer.succeed(None).addCallback(lambda x: (REQ_FILE_ACTN_COMPLETED_OK,))
        return defer.fail(None)

    def connectionLost(self, reason):
        super().connectionLost(reason)
        log.msg("FTP: Lost connection. Reason:", reason)
        if self.addr in self.factory.cnxs.keys():
            self.factory.cnxs[self.addr][0].transport.loseConnection()
            del self.factory.cnxs[self.addr]


class InMemoryFTPFactory(FTPFactory):
    protocol = InMemoryFTP

    def __init__(self, portal=None, userAnonymous="anonymous"):
        super(InMemoryFTPFactory, self).__init__(portal, userAnonymous)
        self.storage = {}  # No sync on storage needed as twisted is single-threaded
        self.decoded = {}
        self.n_items = {}
        self.cnxs = {}  # ip,port -> conn, [paths]
        self.listener_factory = NotifierFactory(self)

    def buildProtocol(self, addr):
        res = super().buildProtocol(addr)
        res.wrappedProtocol.addr = (addr.host, addr.port)
        return res

    def connect_to_listener(self, host, port):
        reactor.callLater(0, reactor.connectTCP, host, port, self.listener_factory)


class InMemoryFTPClient(FTPClient):

    def __init__(self, username='anonymous', password=''):
        super().__init__(username, password, passive=0)

    def peekFile(self, path, protocol):
        cmds = ["PEEK " + self.escapePath(path)]
        return self.receiveFromConnection(cmds, protocol)

    def appendFile(self, path):
        cmds = ["APPD " + self.escapePath(path)]
        return self.sendToConnection(cmds)

    def list(self, protocol=None, path=''):
        cmds = ["LIST " + self.escapePath(path)]
        return self.receiveFromConnection(cmds, protocol)

    def listen(self, path):
        return self.queueStringCommand("LSTN " + self.escapePath(path))

    def ignore(self, path):
        return self.queueStringCommand("NLSN " + self.escapePath(path))


class InMemoryFTPRealm(FTPRealm):

    def __init__(self):
        super().__init__('.')

    def getHomeDirectory(self, avatarId):
        return

    def requestAvatar(self, avatarId, mind, *interfaces):
        for iface in interfaces:
            if iface is IFTPShell:
                avatar = InMemoryFTPShell()
                return IFTPShell, avatar, getattr(avatar, "logout", lambda: None)
        raise NotImplementedError("Only IFTPShell interface is supported by this realm")


def run_server(port: int, start_reactor: bool = True):
    """Starts the tensorshare server, listening to requests coming in on the specified port.

    If :code:`start_reactor = True`, this function is blocking.
    Must be called from the main thread in either case.

    Args:
        port (int): the port on which to listen
        start_reactor (bool): If true, starts the reactor, which means that this function is blocking. Set it to false if you want to run a client in the same process, e.g. for testing purposes.
    """
    # Set up authentication
    p = Portal(InMemoryFTPRealm(), [AllowAnonymousAccess()])
    # Listen
    reactor.listenTCP(port, InMemoryFTPFactory(p))
    if start_reactor:
        reactor.run()


class BufferingProtocol(Protocol):
    """Simple utility class that holds all data written to it in a buffer."""

    def __init__(self):
        self.buffer = io.BytesIO()

    def dataReceived(self, data):
        self.buffer.write(data)


class Notifier(LineReceiver):

    def __init__(self, factory, addr):
        self.factory = factory
        self.addr = addr

    def sendLine(self, line: str):
        super().sendLine(line.encode('ascii'))

    def connectionMade(self):
        if self.addr in self.factory.cnxs.keys():
            self.factory.cnxs[self.addr] = (self, self.factory.cnxs[self.addr][1])  # Replace conn

        else:
            self.factory.cnxs[self.addr] = (self, [])   # Init conn and paths

        log.msg("Notifier connection made. Conns: {}".format(self.factory.cnxs))

    def connectionLost(self, reason):
        if self.addr in self.factory.cnxs.keys():
            del self.factory.cnxs[self.addr]
        log.msg("Notifier connection lost. Conns: {}, reason:".format(self.factory.cnxs), reason)


class NotifierFactory(Factory):

    def __init__(self, factory):
        self.factory = factory

    def buildProtocol(self, addr):
        return Notifier(self.factory, (addr.host, addr.port))

    def startedConnecting(self, connector):
        log.msg('NotifyListenerFactory: Started to connect.')

    def clientConnectionLost(self, connector, reason):
        log.msg('NotifyListenerFactory: Lost connection.  Reason:', reason)

    def clientConnectionFailed(self, connector, reason):
        log.msg('NotifyListenerFactory: Connection failed. Reason:', reason)


class NotifyListener(LineReceiver):

    def __init__(self, factory):
        self.factory = factory
        self.factory.conn = self

    def lineReceived(self, line):
        line = line.decode('ascii')
        items = line.split('|')
        num = int(items[-1])
        path = '|'.join(items[:-1])
        self.factory.notify((path, num))

    def connectionMade(self):
        log.msg("NotifyListener: connected")

    def connectionLost(self, reason):
        log.msg("NotifyListener: connection lost.")


class NotifyListenerFactory(Factory):

    def __init__(self, update_callback: callable):
        self.callback = update_callback  # str -> None
        self.conn = None
        self._port = None

    def get_port(self) -> Optional[int]:
        if self._port is not None:
            return self._port.getHost().port
        return None

    def quit(self):
        if self.conn is not None:
            self.conn.transport.loseConnection()

    def notify(self, x):
        self.callback(x)

    def buildProtocol(self, addr):
        return NotifyListener(self)

    def listen(self, port):
        self._port = reactor.listenTCP(port, self)


# ====== Tests stuff ======
def test(ftp_clt):
    def upload(consumer, file_object):
        print("upload")
        FileSender().beginFileTransfer(file_object, consumer).addCallback(lambda _: consumer.finish()).addCallback(
            lambda _: file_object.close())

    def fail(error):
        print(error)

    import time
    print("Stor")
    val = io.BytesIO()
    val.seek(0)
    torch.save(torch.randn(100, 100), val)
    val.seek(0)

    dC, _ = ftp_clt.storeFile('conf_remote.yml')
    dC.addCallback(upload, val).addErrback(fail)
    time.sleep(0.5)

    print("appd")
    val = io.BytesIO()
    val.seek(0)
    torch.save(torch.randn(100, 100), val)
    val.seek(0)

    dC, _ = ftp_clt.appendFile('conf_remote.yml')
    dC.addCallback(upload, val).addErrback(fail)
    time.sleep(0.5)

    print("List")
    buff = BufferingProtocol()
    dC = ftp_clt.list(buff)
    dC.addCallback(lambda x: print('list result:', buff.buffer.getvalue().decode('ascii'))).addErrback(fail)
    time.sleep(0.5)

    print("Retr")
    buffer = BufferingProtocol()
    dC = ftp_clt.retrieveFile('conf_remote.yml', buffer)

    def cbRetr(x):
        buffer.buffer.seek(0)
        print('retr result', torch.load(buffer.buffer))

    dC.addCallback(cbRetr).addErrback(fail)
    time.sleep(0.5)

    ftp_clt.removeFile('conf_remote.yml')
    time.sleep(0.5)

    buff2 = BufferingProtocol()
    dC = ftp_clt.list(buff2)
    dC.addCallback(lambda x: print('list result:', buff2.buffer.getvalue().decode('ascii'))).addErrback(fail)


def run_client(host='localhost', port=8000, username='anonymous', password=''):
    creator = ClientCreator(reactor, InMemoryFTPClient, username, password)
    creator.connectTCP(host, port).addCallback(test).addErrback(lambda x: print(x))
    reactor.run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--client', action='store_true')
    parser.set_defaults(server=False, client=False)
    args = parser.parse_args()
    assert args.server or args.client
    if args.server:
        run_server(8000)
    if args.client:
        run_client(port=8000)
