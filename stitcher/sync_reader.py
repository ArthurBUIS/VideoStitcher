"""FPS-aware paired frame reader: handles two streams with different FPS."""


# If two FPS values are within this fractional tolerance, treat them as equal
# and skip the desync logic entirely.
DESYNC_TOLERANCE = 0.005   # 0.5%


class FrameSyncReader:
    """
    Reads paired frames from two cv2.VideoCapture objects whose nominal
    FPS may differ. The slower stream is the "driver" (one frame per
    output tick); the faster stream is the "follower" (advance to the
    closest-in-time frame, drop intermediates).

    If the two FPS values match within DESYNC_TOLERANCE, falls through
    to a plain lockstep read with zero overhead.

    Usage:
        reader = FrameSyncReader(cap_a, cap_b, fps_a, fps_b)
        print(reader.summary())
        while True:
            ok, frame_a, frame_b = reader.read()
            if not ok:
                break
            ...
        print(reader.summary_post())   # optional: drop counts

    Properties exposed:
        output_fps : the FPS to use for the output video writer.
    """

    def __init__(self, cap_a, cap_b, fps_a, fps_b):
        self.cap_a = cap_a
        self.cap_b = cap_b
        self.fps_a = float(fps_a)
        self.fps_b = float(fps_b)

        # Detect mismatch.
        if self.fps_a <= 0 or self.fps_b <= 0:
            raise RuntimeError(f"Invalid FPS values: A={self.fps_a}, B={self.fps_b}")

        rel_diff = abs(self.fps_a - self.fps_b) / max(self.fps_a, self.fps_b)
        self.desync_active = rel_diff > DESYNC_TOLERANCE

        if self.desync_active:
            # Driver = slower (we read 1:1 from it). Follower = faster.
            if self.fps_a < self.fps_b:
                self._driver_label = "A"
                self._follower_label = "B"
                self._fps_driver = self.fps_a
                self._fps_follower = self.fps_b
            else:
                self._driver_label = "B"
                self._follower_label = "A"
                self._fps_driver = self.fps_b
                self._fps_follower = self.fps_a
        else:
            self._driver_label = None
            self._follower_label = None
            self._fps_driver = min(self.fps_a, self.fps_b)
            self._fps_follower = max(self.fps_a, self.fps_b)

        self.output_fps = self._fps_driver

        # Counters (used for stats and for the matching arithmetic).
        self._driver_idx = 0          # next driver index to read
        self._follower_idx = 0        # next follower index to read
        self._dropped_count = 0       # number of follower frames discarded

    def summary(self):
        if self.desync_active:
            return (f"[sync] FPS mismatch detected: A={self.fps_a:.3f}, "
                    f"B={self.fps_b:.3f} (diff={100*abs(self.fps_a-self.fps_b)/max(self.fps_a, self.fps_b):.2f}%). "
                    f"Driver={self._driver_label} ({self._fps_driver:.3f} fps), "
                    f"Follower={self._follower_label} ({self._fps_follower:.3f} fps). "
                    f"Output FPS = {self.output_fps:.3f}.")
        else:
            return (f"[sync] FPS match (A={self.fps_a:.3f}, B={self.fps_b:.3f}). "
                    f"Lockstep read. Output FPS = {self.output_fps:.3f}.")

    def summary_post(self):
        if self.desync_active:
            return (f"[sync] Read {self._driver_idx} driver frames from "
                    f"{self._driver_label}, {self._follower_idx} follower frames "
                    f"from {self._follower_label}, dropped {self._dropped_count} "
                    f"follower frames to maintain temporal alignment.")
        else:
            return (f"[sync] Read {self._driver_idx} pairs in lockstep "
                    f"(no frames dropped).")

    def read(self):
        """
        Returns (ok, frame_a, frame_b).
        ok is False when either stream runs out.
        """
        if not self.desync_active:
            ok_a, fa = self.cap_a.read()
            ok_b, fb = self.cap_b.read()
            if not (ok_a and ok_b):
                return False, None, None
            self._driver_idx += 1
            self._follower_idx += 1
            return True, fa, fb

        # Desync path.
        # 1. Read the next driver frame.
        cap_driver = self.cap_a if self._driver_label == "A" else self.cap_b
        cap_follower = self.cap_b if self._driver_label == "A" else self.cap_a
        ok_d, frame_driver = cap_driver.read()
        if not ok_d:
            return False, None, None

        # The driver frame's timestamp is (driver_idx) / fps_driver.
        # Note: we use driver_idx BEFORE incrementing, because the frame
        # we just read corresponds to that index.
        driver_t = self._driver_idx / self._fps_driver
        self._driver_idx += 1

        # 2. Advance the follower to the frame whose timestamp is closest
        # to driver_t. The follower's frame i has timestamp i / fps_follower.
        # We want the integer i minimizing |i/fps_follower - driver_t|, i.e.
        # the round() of (driver_t * fps_follower).
        target_follower_idx = int(round(driver_t * self._fps_follower))
        # We must read at least once (can't skip the current frame and then
        # not consume any), and we must consume exactly enough to reach
        # target_follower_idx + 1 (i.e., next position is target+1, last
        # consumed is target).
        # _follower_idx is the index of the NEXT frame to read.
        # We need to read frames until we've consumed index target_follower_idx.
        while self._follower_idx < target_follower_idx:
            ok_f, _ = cap_follower.read()
            if not ok_f:
                return False, None, None
            self._follower_idx += 1
            self._dropped_count += 1
        # Now read the target frame itself.
        ok_f, frame_follower = cap_follower.read()
        if not ok_f:
            return False, None, None
        self._follower_idx += 1

        # 3. Map back to (frame_a, frame_b) regardless of which is driver.
        if self._driver_label == "A":
            return True, frame_driver, frame_follower
        else:
            return True, frame_follower, frame_driver
