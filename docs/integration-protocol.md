# VideoStitcher ↔ Electron Integration Protocol

**Status:** Draft v0. Subject to change as the first end-to-end
integration spike runs. Once the protocol is exercised by real code
on both sides, this document graduates from draft and the version
number in §9 governs further changes.

**Scope:** This document specifies how the Electron app (the
portals-projector-agent product) and VideoStitcher (this Python
stitching service) talk to each other when they run on the same
machine. It covers:

- How they establish a connection (named pipes)
- What bytes they send each other (frames + control commands)
- What happens on errors, slow consumers, and shutdown

It does **not** describe what either side does internally. Both
sides are free to change their implementations as long as they
continue to honor this contract.


## 1. Why this document exists

VideoStitcher and the Electron product are separate codebases with
separate release cadences. The integration boundary between them
needs to be precise enough that:

- Either side can be changed independently as long as the contract
  holds.
- Misalignment (e.g. one side expecting 4-byte timestamps while the
  other sends 8-byte) produces a clear error at handshake instead
  of cryptic crashes mid-stream.
- A new contributor on either side can understand the boundary
  without reading the other codebase.

This is the same role a USB specification plays for device makers
and host OS authors -- they don't share source code, they share a
spec.


## 2. The two channels

Two separate named pipes are used. Both are created by the Electron
side on session start; Python connects as a client.

```
\\.\pipe\portals-stitcher-<session>-control     ← JSON messages
\\.\pipe\portals-stitcher-<session>-frames      ← binary frame data
```

`<session>` is a random string the Electron side generates per
launch (a UUID is fine). Per-launch uniqueness avoids stale-pipe
collisions after crashes.

**Why two channels?** A single channel would force both sides to
multiplex frames and commands together. Two channels means each
side reads from a dedicated stream and the formats are independent:
JSON-text on one, binary on the other. Cleaner parsers, easier
debugging.


## 3. Lifecycle

```
1. Electron generates a session ID and creates both pipes.
2. Electron spawns the Python process with
       --io pipe --session <id>
   plus any usual VideoStitcher flags (weights, devices, etc.).
3. Python connects to both pipes (control first, then frames).
4. Both sides exchange `hello` / `hello_ack` on the control channel.
5. Electron sends `start_session` with calibration + input camera
   metadata.
6. Python responds with `session_started` once models are loaded
   and ready.
7. Frame data flows in both directions on the frames channel:
      Electron → Python: paired camera frames
      Python  → Electron: stitched output frames
8. Either side can send `stop_session` on the control channel.
9. Python flushes any in-flight frames and responds with
   `session_stopped`.
10. Process exits on a follow-up `shutdown` or signal.
```

A Python process can serve multiple `start_session` /
`stop_session` cycles in a single lifetime -- model weights load
once and are reused.


## 4. Control channel format

Line-delimited JSON. UTF-8 encoded. One message per line. Both
sides parse a line at a time.

### From Electron to Python

```json
{ "type": "hello", "protocol_versions": [1] }

{
  "type": "start_session",
  "calibration": { ... },
  "input_cameras": [
    { "index": 0, "label": "left",  "width": 1920, "height": 1080 },
    { "index": 1, "label": "right", "width": 1920, "height": 1080 }
  ]
}

{ "type": "stop_session" }

{ "type": "shutdown" }
```

The exact shape of `calibration` is TBD; it carries whatever a
recipient needs to reconstruct the homographies + canvas geometry
without recomputing them from frame 0. For the first spike, this
may simply be empty (Python re-runs its existing calibration on
the first incoming frames).

### From Python to Electron

```json
{ "type": "hello_ack", "protocol_version": 1, "stitcher_version": "1.2.3" }

{ "type": "session_started", "output_width": 1080, "output_height": 720 }

{ "type": "session_stopped" }

{ "type": "error", "code": "...", "message": "..." }

{ "type": "log", "level": "info|warn|error", "message": "..." }
```

**Why JSON for control?** Control messages are rare (a dozen per
session at most). Performance doesn't matter; debuggability does.
JSON shows up readably in logs and Wireshark-style captures.


## 5. Frames channel format

Binary. Every frame is a fixed-size 32-byte header followed by raw
pixel bytes. Little-endian for every multi-byte integer.

```
offset  size  field             type        meaning
------  ----  ----------------  ----------  ----------------------------------
   0    4     magic             ASCII       "FRMV" -- sanity check, detects desync
   4    4     camera_index      uint32      0 = left, 1 = right
                                            255 = stitched output (Python → Electron)
   8    8     timestamp_us      uint64      microseconds, monotonic clock
                                            sourced from Electron's performance.now()
  16    4     width             uint32      pixels
  20    4     height            uint32      pixels
  24    4     format            uint32      1 = RGBA8888 (only format defined in v1)
  28    4     payload_length    uint32      = width * height * 4 for RGBA8888
  32    N     payload           bytes       raw pixels, row-major, top-down
```

**Why this layout?**

- Fixed-size header → the reader always knows exactly how many bytes
  to consume before the next frame.
- Magic bytes → catches "we got out of sync" bugs immediately
  instead of decoding garbage.
- Explicit width / height / format → no global state to drift; every
  frame self-describes.
- Timestamp comes from Electron → Python never has to know what
  "now" is.

**Why RGBA8888?** It's what Electron's `ImageData` produces
natively, and what NumPy can index as `(H, W, 4) uint8` without any
conversion. Compact alternatives (JPEG, NV12 YUV) would require
encoding work on one side and decoding on the other, wasting CPU
for no real bandwidth benefit on a loopback transport.


## 6. Frame flow

**Input (Electron → Python):**

- Electron sends an interleaved stream of frames from camera 0 and
  camera 1.
- Frames carry their own capture timestamp. Electron is responsible
  for stamping them coherently across the two cameras (same clock
  source).
- Python pairs them up internally (using its existing
  `FrameSyncReader` logic; same idea, different source).

**Output (Python → Electron):**

- Python emits one stitched frame for each pair of input frames it
  consumes.
- Camera index in the output header is `255` (sentinel value).


## 7. Backpressure and dropping

Realtime video is unforgiving: a frame from 2 seconds ago is
useless. Buffering frames forever just hides slowness and produces
unbounded memory growth. Both sides drop instead.

**If Python is slow** (compute can't keep up with input rate):

- Electron monitors its send buffer per camera.
- When the buffer reaches N frames, the oldest unsent frame is
  discarded.
- A `log` control message is sent describing the drop ("dropped
  cam=0 ts=12345").
- Suggested N: **3 frames** per camera (~100 ms at 30 fps).

**If Electron is slow consuming output:**

- Python applies the same logic in reverse.
- Same suggested N (3 frames, ~100 ms).

The Electron side decides whether to count drops as an actionable
quality signal.


## 8. Errors and shutdown

Any unexpected condition on either side:

1. The detecting side sends an `error` control message describing
   what happened.
2. Both pipes are closed.
3. The process that detected the error exits.

Specific cases:

- Python fails to load YOLOE / depth weights → emit
  `error("models_failed", ...)`, exit.
- Pipe write fails (`EPIPE` / peer disconnected) → no message
  possible, just exit.
- Frame header magic doesn't match `"FRMV"` → emit
  `error("frame_desync", ...)`, exit.
- Frame payload length mismatches what the header promised → emit
  `error("frame_length", ...)`, exit.

The Electron supervisor decides whether to restart Python after an
error. This protocol doesn't prescribe a restart policy.


## 9. Versioning

The protocol carries a single integer version. The current version
is **1**.

**Bump rules:**

- Add a new field to an existing control message → no bump
  (backward compatible; old peers ignore unknown fields).
- Add a new control message type → no bump (old peers respond with
  an `error("unknown_message")` if they receive it).
- Change the frame header layout, change the meaning of an existing
  field, or change the payload pixel format defaults → **bump**.

**Negotiation:** Electron advertises the versions it can speak in
the `hello` message; Python picks the highest one it can also
speak and announces it in `hello_ack`. Mismatch (no shared version)
→ Python emits `error("protocol_version")` and exits.


## 10. Design decisions worth recording

These are choices made during the v0 draft. Recording them here so
later readers know why they're the way they are.

- **Calibration travels in `start_session`** rather than being
  loaded from a file at Python startup. This lets the Electron app
  carry per-room or per-portal calibration without Python needing
  to know about filesystem layout. The first spike may pass an
  empty calibration and let Python use its current frame-0
  computation; the protocol slot is there for when persistent
  calibration is wired up.

- **Output is a single stitched RGBA frame** rather than tiled or
  encoded chunks. Symmetric with input format; same RGBA buffer is
  blittable directly to a renderer canvas with no decoding.

- **Two pipes, not one.** Costs marginally more setup; saves a lot
  of parser complexity. See §2.

- **JSON for control, binary for frames.** Control volume is tiny;
  frame volume is huge. Pick the right format for each.

- **Electron pairs frames before sending.** Python could pair on
  its side, but Electron is closer to the capture clock and has
  better timestamp accuracy. Putting pairing on the Electron side
  also means Python's input stream is simpler (one frame at a
  time).


## 11. Implementation notes (post-spike)

Lessons from the first end-to-end protocol spike
(`tools/test_pipe_harness.py`) that aren't strictly part of the
wire format but bind anyone implementing either side:

- **Concurrent send and receive on both ends.** Each side MUST run
  the input-send loop and the output-read loop on independent
  threads (or independent async tasks). A single-threaded
  "send-all-inputs then read-all-outputs" pattern *deadlocks*:
  typical TCP send buffers are ~64 KB on Windows, so after the
  first 1080p RGBA frame (~8 MB) both sides' `sendall` calls block
  waiting for the peer to drain. The harness's "host" side hit
  this and was restructured to drain stitched output on a reader
  thread while the input sender runs on the main thread.

- **Pair frames on the sender side.** Each frame header carries
  its own timestamp; PipeFrameSource pairs them by waiting for one
  frame per camera before emitting a tuple. Senders should still
  alternate cameras 0, 1, 0, 1, ... at roughly the same wall-clock
  rate to keep pairing latency low.

- **One transport per channel.** The two named pipes (control,
  frames) are *not* multiplexed onto a single connection. Cleaner
  parsers; easier debugging; one side's slowness on one channel
  doesn't block the other.


## 12. Open items (post-spike)

To be resolved once we have a working spike and concrete data:

- The exact shape of `calibration` in `start_session`.
- Whether `stitcher_version` in `hello_ack` should map to a release
  tag of the VideoStitcher repo, a semver, or something else.
- Whether output frames should optionally carry seam-debug overlays
  (current debug-mask, debug-seam) when Electron opts in.
- Whether session restart needs an explicit `reset` message or
  whether `stop_session` + `start_session` is sufficient.
