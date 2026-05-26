"""
Transport layer for the integration protocol.

Named pipes (per docs/integration-protocol.md §2) are the production
transport. For the development spike we use TCP on localhost because
Python's stdlib `socket` module works the same on every OS with no
extra dependencies, and the *bytes* on the wire are identical to
what named pipes will carry -- only the connection-setup code
differs. A future NamedPipeTransport (via pywin32) can be added
without touching the protocol code.

Two roles:
  Transport       a connected, bidirectional byte stream (used by
                  PipeFrameSource / PipeFrameSink / ControlChannel).
  TCPListener     server-side helper: opens a listening socket,
                  yields a Transport when a client connects.

The Python (VideoStitcher) side always plays the CLIENT role: it
connects to addresses the Electron host has already opened. The
test harness uses TCPListener to play the Electron role.
"""

import socket
from abc import ABC, abstractmethod


class Transport(ABC):
    """A bidirectional byte stream."""

    @abstractmethod
    def read_exact(self, n):
        """Read exactly n bytes. Raises ConnectionError on disconnect."""

    @abstractmethod
    def read_line(self):
        """
        Read until the next b'\\n' (inclusive). Returns the line bytes.
        Raises ConnectionError on disconnect mid-line.
        """

    @abstractmethod
    def write(self, data):
        """Write all bytes; block until done."""

    @abstractmethod
    def close(self):
        """Close the connection. Safe to call multiple times."""


class TCPTransport(Transport):
    """
    TCP-on-localhost transport, client-side. Constructed by
    connecting to (host, port). Used by PipeFrameSource /
    PipeFrameSink / ControlChannel on the Python stitcher side.
    """

    def __init__(self, host, port, timeout=None):
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Set blocking I/O regardless of the connect timeout above.
        sock.settimeout(None)
        self._sock = sock
        self._buf = bytearray()
        self._closed = False

    # Internal constructor for server-accepted sockets.
    @classmethod
    def _from_accepted_socket(cls, sock):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(None)
        obj = cls.__new__(cls)
        obj._sock = sock
        obj._buf = bytearray()
        obj._closed = False
        return obj

    def read_exact(self, n):
        if self._closed:
            raise ConnectionError("transport closed")
        # First, drain anything sitting in the line buffer (a mixed-
        # mode caller may have buffered bytes from a previous read).
        out = bytearray()
        if self._buf:
            take = min(n, len(self._buf))
            out.extend(self._buf[:take])
            del self._buf[:take]
        while len(out) < n:
            try:
                chunk = self._sock.recv(n - len(out))
            except OSError as e:
                raise ConnectionError(str(e)) from e
            if not chunk:
                raise ConnectionError(
                    f"transport closed after {len(out)}/{n} bytes"
                )
            out.extend(chunk)
        return bytes(out)

    def read_line(self):
        if self._closed:
            raise ConnectionError("transport closed")
        while b"\n" not in self._buf:
            try:
                chunk = self._sock.recv(4096)
            except OSError as e:
                raise ConnectionError(str(e)) from e
            if not chunk:
                if self._buf:
                    raise ConnectionError("transport closed mid-line")
                raise ConnectionError("transport closed")
            self._buf.extend(chunk)
        idx = self._buf.index(b"\n") + 1
        line = bytes(self._buf[:idx])
        del self._buf[:idx]
        return line

    def write(self, data):
        if self._closed:
            raise ConnectionError("transport closed")
        try:
            self._sock.sendall(data)
        except OSError as e:
            raise ConnectionError(str(e)) from e

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass


class TCPListener:
    """
    Server-side helper. Opens a listening socket on (host, port) and
    yields a TCPTransport on accept().

    Used by the test harness (tools/test_pipe_harness.py) to play
    the Electron role: the host listens, the stitcher connects.
    In production the Electron app plays this role with Win32 named
    pipes; the protocol bytes are the same.

    Pass port=0 to let the OS pick a free port; read it back via the
    .port property after construction.
    """

    def __init__(self, host="127.0.0.1", port=0, backlog=1):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(backlog)

    @property
    def port(self):
        return self._sock.getsockname()[1]

    def accept(self):
        """Block until a client connects; return a TCPTransport."""
        client_sock, _addr = self._sock.accept()
        return TCPTransport._from_accepted_socket(client_sock)

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass
