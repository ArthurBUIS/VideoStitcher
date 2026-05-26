"""
Wire-format helpers for the integration protocol with the Electron
product. See docs/integration-protocol.md for the spec these helpers
implement.

Pure functions: pack/unpack the binary frame header, build/parse
JSON control messages. No I/O lives here -- the transport (TCP,
named pipes, ...) is separate. Easy to unit-test.
"""

import json
import struct


# Protocol version this implementation speaks. Negotiated during the
# hello / hello_ack handshake.
PROTOCOL_VERSION = 1

# Sentinel camera_index value Python uses for stitched output frames
# (see docs/integration-protocol.md §6).
OUTPUT_CAMERA_INDEX = 255

# Pixel format codes. Only RGBA8888 is defined in protocol v1.
FORMAT_RGBA8888 = 1


# Frame header layout (docs/integration-protocol.md §5):
#   magic           4 bytes (b"FRMV")
#   camera_index    4 bytes uint32 LE
#   timestamp_us    8 bytes uint64 LE
#   width           4 bytes uint32 LE
#   height          4 bytes uint32 LE
#   format          4 bytes uint32 LE
#   payload_length  4 bytes uint32 LE
#   ---------------------------------
#   total           32 bytes
_HEADER_FMT = "<4sIQIIII"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)
assert HEADER_SIZE == 32, (
    f"protocol header should be 32 bytes, got {HEADER_SIZE}"
)

_MAGIC = b"FRMV"


class ProtocolError(Exception):
    """Raised when wire-format bytes don't match the protocol spec."""


def pack_frame_header(camera_index, timestamp_us, width, height,
                      payload_length, fmt=FORMAT_RGBA8888):
    """Pack a frame header into HEADER_SIZE (32) bytes."""
    return struct.pack(
        _HEADER_FMT,
        _MAGIC, camera_index, timestamp_us,
        width, height, fmt, payload_length,
    )


def unpack_frame_header(buf):
    """
    Unpack a HEADER_SIZE-byte frame header. Returns a dict with keys
    'camera_index', 'timestamp_us', 'width', 'height', 'format',
    'payload_length'.

    Raises ProtocolError if `buf` is the wrong length or if the magic
    bytes don't match (catches "we got out of sync" failures
    immediately rather than producing garbage).
    """
    if len(buf) != HEADER_SIZE:
        raise ProtocolError(
            f"header length {len(buf)} != {HEADER_SIZE}"
        )
    magic, cam, ts, w, h, fmt, plen = struct.unpack(_HEADER_FMT, buf)
    if magic != _MAGIC:
        raise ProtocolError(
            f"frame_desync: expected magic {_MAGIC!r}, got {magic!r}"
        )
    return {
        "camera_index": int(cam),
        "timestamp_us": int(ts),
        "width": int(w),
        "height": int(h),
        "format": int(fmt),
        "payload_length": int(plen),
    }


# ---------------------------------------------------------------------------
# Control channel helpers (line-delimited JSON, UTF-8)
# ---------------------------------------------------------------------------


def encode_control_message(msg):
    """Serialize a control message dict to bytes (one UTF-8 JSON line)."""
    return (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")


def decode_control_message(line):
    """
    Parse one UTF-8 JSON line into a dict. Trailing newline is OK.
    Returns None on empty/whitespace-only input.
    Raises ProtocolError on malformed JSON.
    """
    text = line.decode("utf-8").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"control_decode: {e}") from e
    if not isinstance(obj, dict):
        raise ProtocolError(
            f"control_decode: expected JSON object, got {type(obj).__name__}"
        )
    return obj


# ---------------------------------------------------------------------------
# Control message factories
# ---------------------------------------------------------------------------


def make_hello(versions=(PROTOCOL_VERSION,)):
    return {"type": "hello", "protocol_versions": list(versions)}


def make_hello_ack(stitcher_version="unknown"):
    return {
        "type": "hello_ack",
        "protocol_version": PROTOCOL_VERSION,
        "stitcher_version": stitcher_version,
    }


def make_start_session(input_cameras, calibration=None):
    return {
        "type": "start_session",
        "calibration": calibration or {},
        "input_cameras": list(input_cameras),
    }


def make_session_started(output_width, output_height):
    return {
        "type": "session_started",
        "output_width": int(output_width),
        "output_height": int(output_height),
    }


def make_stop_session():
    return {"type": "stop_session"}


def make_session_stopped():
    return {"type": "session_stopped"}


def make_shutdown():
    return {"type": "shutdown"}


def make_error(code, message):
    return {"type": "error", "code": code, "message": message}


def make_log(level, message):
    return {"type": "log", "level": level, "message": message}
