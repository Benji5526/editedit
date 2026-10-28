"""The speed-ups must not change results: parallel SAM 2 frame loading, prefetching reader, unbuffered frame reads, copy-free writes."""
import os
import threading
import time

import cv2
import numpy as np
import pytest

from helpers import H, W, draw_frame, make_video, read_frame_code
from src.video_io import FrameReader, PrefetchReader, SegmentWriter, VideoError, probe_video


# ------------------------------------------------------------------ FrameReader
def test_frames_are_independent_writable_arrays(video):
    info = probe_video(video)
    with FrameReader(info, max_frames=3) as r:
        a, b, c = r.read(), r.read(), r.read()
        assert r.read() is None and r.read() is None                      # end of stream is stable
    for f in (a, b, c):
        assert f.flags.writeable and f.shape == (H, W, 3) and f.dtype == np.uint8
    codes = [read_frame_code(x) for x in (a, b, c)]
    assert codes == [0, 1, 2]
    a[:] = 0                                                              # mutating one frame must not touch the others
    assert read_frame_code(b) == 1 and read_frame_code(c) == 2


def test_truncated_stream_raises(video):
    info = probe_video(video)
    r = FrameReader(info, max_frames=2)
    real = r._p.stdout

    class Cut:
        """delivers only half of the first frame, then end of file"""
        def __init__(self):
            self.sent = False

        def readinto(self, mv):
            if self.sent:
                return 0
            self.sent = True
            n = len(mv) // 2
            mv[:n] = b"\x01" * n
            return n

        def close(self):
            real.close()

    r._p.stdout = Cut()
    with pytest.raises(VideoError, match="truncated"):
        r.read()
    r.close()


# ------------------------------------------------------------------ PrefetchReader
def test_prefetch_gives_the_same_frames_in_the_same_order(video):
    info = probe_video(video)
    with PrefetchReader(FrameReader(info), depth=3) as r:
        got = [read_frame_code(f) for f in r]
        assert r.read() is None
    assert got == list(range(160))


def test_prefetch_with_start_frame_and_scaling(video):
    info = probe_video(video)
    with PrefetchReader(FrameReader(info, start_frame=77, max_frames=5, scale_to=(320, 180))) as r:
        fs = list(r)
    assert len(fs) == 5 and fs[0].shape == (180, 320, 3)


class FakeReader:
    out_w, out_h = 4, 4
    frames_read = 0

    def __init__(self, n, fail_at=None, delay=0.0):
        self.n, self.fail_at, self.delay, self.reads, self.closed = n, fail_at, delay, 0, False

    def read(self):
        if self.delay:
            time.sleep(self.delay)
        if self.fail_at is not None and self.reads == self.fail_at:
            raise VideoError("decoder died")
        if self.reads >= self.n:
            return None
        self.reads += 1
        return np.full((4, 4, 3), self.reads, np.uint8)

    def close(self):
        self.closed = True


def test_prefetch_delivers_a_read_error_at_the_right_position():
    r = PrefetchReader(FakeReader(10, fail_at=3), depth=2)
    assert [int(r.read()[0, 0, 0]) for _ in range(3)] == [1, 2, 3]       # frames before the failure arrive first
    with pytest.raises(VideoError, match="decoder died"):
        r.read()
    assert r.read() is None                                              # then the stream is over
    r.close()


def test_prefetch_reads_ahead_by_a_bounded_amount_and_stops_cleanly():
    fake = FakeReader(1000)
    r = PrefetchReader(fake, depth=3)
    deadline = time.time() + 5
    while fake.reads < 4 and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    assert fake.reads <= 3 + 2                                           # queue depth + the frame in hand + one blocked put
    assert int(r.read()[0, 0, 0]) == 1
    t0 = time.time()
    r.close()                                                            # consumer leaves early with a full queue: no hang
    assert time.time() - t0 < 3 and fake.closed
    assert not any(t.name == "PrefetchReader" and t.is_alive() for t in threading.enumerate())


def test_prefetch_close_while_the_reader_is_blocked():
    class Blocking(FakeReader):
        def __init__(self):
            super().__init__(10)
            self.release = threading.Event()

        def read(self):
            self.release.wait(5)
            return None

        def close(self):
            self.closed = True
            self.release.set()                                           # like killing ffmpeg: unblocks the pending read

    fake = Blocking()
    r = PrefetchReader(fake)
    time.sleep(0.1)
    t0 = time.time()
    r.close()
    assert time.time() - t0 < 3 and fake.closed


# ------------------------------------------------------------------ writer
def test_writer_accepts_non_contiguous_frames(video, tmp_path):
    info = probe_video(video)
    out = str(tmp_path / "flip.mp4")
    w = SegmentWriter(out, info, "libx264", 18, "ultrafast")
    for t in range(5):
        w.write(draw_frame(t)[:, ::-1])                                  # a reversed view: negative strides, not C-contiguous
    w.close()
    with FrameReader(probe_video(out)) as r:
        first = r.read()
    expect = draw_frame(0)[:, ::-1]
    assert np.abs(first.astype(int) - expect.astype(int)).mean() < 12       # lossy codec, but clearly the flipped picture
    assert np.abs(first.astype(int) - draw_frame(0).astype(int)).mean() > 25


# ------------------------------------------------------------------ SAM 2 parallel frame loading
def test_parallel_frame_loader_equals_the_original(tmp_path):
    torch = pytest.importorskip("torch")
    from sam2.utils import misc
    from src.tracking import parallel_load_video_frames

    d = tmp_path / "frames"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(23):                                                  # 0..22: sorting must be numeric (2 before 10)
        img = (rng.random((200, 300, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(d / f"{i}.jpg"), img)
    for offload in (True, False):
        a = misc.load_video_frames(str(d), 1024, offload, async_loading_frames=False, compute_device=torch.device("cuda"))
        b = parallel_load_video_frames(str(d), 1024, offload, async_loading_frames=False, compute_device=torch.device("cuda"))
        assert a[1:] == b[1:] == (200, 300)
        assert a[0].shape == b[0].shape == (23, 3, 1024, 1024) and a[0].device == b[0].device
        assert torch.equal(a[0], b[0])                                    # bit-identical tensors


@pytest.mark.parametrize("inference", [True, False])
def test_parallel_frame_loader_works_inside_and_outside_inference_mode(tmp_path, inference):
    """SAM 2 calls the loader inside torch.inference_mode(); the worker threads must cope with the resulting inference tensor
    (regression: 'Inplace update to inference tensor outside InferenceMode' made every chunk fail with model_error)."""
    torch = pytest.importorskip("torch")
    from sam2.utils import misc
    from src.tracking import parallel_load_video_frames

    d = tmp_path / "f"
    d.mkdir()
    for i in range(12):
        cv2.imwrite(str(d / f"{i}.jpg"), (np.random.default_rng(i).random((90, 160, 3)) * 255).astype(np.uint8))
    with torch.inference_mode(inference):
        a = misc.load_video_frames(str(d), 256, True, async_loading_frames=False, compute_device=torch.device("cuda"))
        b = parallel_load_video_frames(str(d), 256, True, async_loading_frames=False, compute_device=torch.device("cuda"), workers=4)
        assert b[0].is_inference() == a[0].is_inference() == inference
    assert torch.equal(a[0], b[0]) and a[1:] == b[1:]


def test_parallel_loader_delegates_what_it_does_not_handle(tmp_path):
    pytest.importorskip("torch")
    from src.tracking import parallel_load_video_frames
    with pytest.raises(RuntimeError, match="no images"):
        empty = tmp_path / "empty"
        empty.mkdir()
        parallel_load_video_frames(str(empty), 256, True)


def test_the_tracker_uses_the_parallel_loader():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from src.tracking import SAM2Tracker, parallel_load_video_frames
    import sam2.sam2_video_predictor as svp
    tr = SAM2Tracker(None, "configs/sam2.1/sam2.1_hiera_t.yaml", device="cuda", _random_weights_api_test=True)
    assert tr.parallel_loading and svp.load_video_frames is parallel_load_video_frames
