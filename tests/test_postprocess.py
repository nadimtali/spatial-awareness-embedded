"""Tests for the detection pre/post-processing shared by OnnxDetector and DpuDetector.

Runs without a board and without torch/ultralytics (stdlib unittest):

    python -m unittest discover -s tests -v

DpuDetector is exercised through a fake VART runner. The last test case
cross-checks the YOLOv8 head decoding against the real model and is skipped
unless torch, ultralytics and yolov8n.pt are available (dev machine only).
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detector import (  # noqa: E402
    MAX_DET,
    DpuDetector,
    decode_predictions,
    letterbox,
    nms,
    yolov8_raw_to_pred,
)

REG_MAX = 16
NBOX = 4 * REG_MAX


class TestLetterbox(unittest.TestCase):
    def test_landscape_frame_is_padded_top_and_bottom(self):
        frame = np.zeros((576, 768, 3), np.uint8)
        frame[288:, :] = 255  # bottom half white
        img, gain, pad = letterbox(frame, 640)
        self.assertEqual(img.shape, (640, 640, 3))
        self.assertAlmostEqual(gain, 640 / 768)
        self.assertEqual(pad, (0, 80))            # 576 * 0.8333 = 480 -> 80 px each side
        self.assertTrue((img[:80] == 114).all())  # pad value
        self.assertTrue((img[-80:] == 114).all())
        self.assertTrue((img[80:315] == 0).all())  # top half of the image
        self.assertTrue((img[325:560] == 255).all())  # bottom half

    def test_square_frame_has_no_padding(self):
        img, gain, pad = letterbox(np.zeros((320, 320, 3), np.uint8), 640)
        self.assertEqual((img.shape[:2], gain, pad), ((640, 640), 2.0, (0, 0)))


class TestNms(unittest.TestCase):
    def test_suppresses_overlap_and_keeps_far_box(self):
        boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], np.float32)
        scores = np.array([0.9, 0.8, 0.7], np.float32)
        self.assertEqual(nms(boxes, scores, 0.7).tolist(), [0, 2])  # IoU(0,1) = 0.82

    def test_keeps_overlap_below_threshold(self):
        boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], np.float32)  # IoU = 1/3
        scores = np.array([0.5, 0.9], np.float32)
        self.assertEqual(nms(boxes, scores, 0.5).tolist(), [1, 0])  # best first

    def test_empty(self):
        self.assertEqual(len(nms(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0.5)), 0)


def pred_row(cx, cy, w, h, cls, score, nc=3):
    row = np.zeros(4 + nc, np.float32)
    row[:4] = cx, cy, w, h
    row[4 + cls] = score
    return row


class TestDecodePredictions(unittest.TestCase):
    NAMES = {0: "person", 1: "car", 2: "dog"}

    def decode(self, rows, conf=0.1, iou=0.7, gain=0.5, pad=(0, 80), frame_hw=(960, 1280)):
        return decode_predictions(np.stack(rows), conf, iou, gain, pad, frame_hw, self.NAMES)

    def test_undoes_letterbox(self):
        # letterboxed box (100, 180)-(200, 380) with gain 0.5, pad_y 80
        d = self.decode([pred_row(150, 280, 100, 200, cls=1, score=0.9)])
        np.testing.assert_allclose(d.boxes, [[200, 200, 400, 600]])
        self.assertEqual(d.labels, ["car"])
        self.assertEqual(d.boxes.dtype, np.float32)

    def test_confidence_filter_is_strict_and_keeps_low_scores(self):
        d = self.decode([pred_row(100, 200, 20, 20, 0, 0.15),   # kept: ByteTrack low stage
                         pred_row(300, 200, 20, 20, 0, 0.10),   # == conf: dropped
                         pred_row(500, 200, 20, 20, 0, 0.05)])  # dropped
        np.testing.assert_allclose(d.scores, [0.15])

    def test_class_with_best_score_wins(self):
        row = pred_row(100, 200, 20, 20, 0, 0.3)
        row[4 + 2] = 0.6
        d = self.decode([row])
        self.assertEqual(d.class_ids.tolist(), [2])
        np.testing.assert_allclose(d.scores, [0.6])

    def test_nms_is_class_aware(self):
        d = self.decode([pred_row(100, 200, 50, 50, 0, 0.9),
                         pred_row(102, 200, 50, 50, 0, 0.8),   # same class, overlapping
                         pred_row(101, 200, 50, 50, 1, 0.7)])  # other class, overlapping
        self.assertEqual(sorted(d.class_ids.tolist()), [0, 1])
        self.assertEqual(len(d), 2)

    def test_clips_to_frame(self):
        d = self.decode([pred_row(5, 85, 40, 40, 0, 0.9)])  # sticks out top-left
        self.assertEqual(d.boxes[0, :2].tolist(), [0, 0])

    def test_caps_detections(self):
        rows = [pred_row(20 * (i % 30) + 10, 20 * (i // 30) + 90, 10, 10, 0, 0.5)
                for i in range(400)]  # non-overlapping grid
        d = self.decode(rows, frame_hw=(2000, 2000))
        self.assertEqual(len(d), MAX_DET)

    def test_no_detections(self):
        d = self.decode([pred_row(100, 200, 20, 20, 0, 0.01)])
        self.assertEqual((len(d), d.boxes.shape), (0, (0, 4)))


def peaked_dfl(l, t, r, b, peak=50.0):
    """DFL logits (64,) whose softmax is ~one-hot at bins l, t, r, b."""
    logits = np.zeros((4, REG_MAX), np.float32)
    for side, k in enumerate((l, t, r, b)):
        logits[side, k] = peak
    return logits.ravel()


def empty_maps(imgsz=640, nc=3, cls_logit=-20.0):
    return [np.concatenate([np.zeros((imgsz // s, imgsz // s, NBOX), np.float32),
                            np.full((imgsz // s, imgsz // s, nc), cls_logit, np.float32)], axis=2)
            for s in (8, 16, 32)]


class TestYolov8RawToPred(unittest.TestCase):
    def test_shape_and_anchor_order(self):
        maps = empty_maps()
        pred = yolov8_raw_to_pred(maps[::-1], 640)  # order of maps must not matter
        self.assertEqual(pred.shape, (80 * 80 + 40 * 40 + 20 * 20, 4 + 3))
        # first anchor: stride 8 cell (0, 0) -> center (4, 4); last: stride 32 cell (19, 19)
        np.testing.assert_allclose(pred[0, :2], [4, 4])
        np.testing.assert_allclose(pred[-1, :2], [19.5 * 32, 19.5 * 32])

    def test_dfl_box_decoding(self):
        maps = empty_maps()
        gy, gx = 1, 2
        maps[0][gy, gx, :NBOX] = peaked_dfl(2, 3, 4, 5)   # stride 8
        pred = yolov8_raw_to_pred(maps, 640)
        row = pred[gy * 80 + gx]
        # anchor (2.5, 1.5) -> x1 0.5, y1 -1.5, x2 6.5, y2 6.5 (grid units) * 8
        np.testing.assert_allclose(row[:4], [28, 20, 48, 64], atol=1e-4)

    def test_uniform_dfl_is_mean_bin(self):
        pred = yolov8_raw_to_pred(empty_maps(), 640)
        # zero logits -> uniform softmax -> distance 7.5 bins per side
        np.testing.assert_allclose(pred[0, 2:4], [15 * 8, 15 * 8], atol=1e-4)
        np.testing.assert_allclose(pred[-1, 2:4], [15 * 32, 15 * 32], atol=1e-3)

    def test_class_sigmoid(self):
        maps = empty_maps(cls_logit=0.0)
        maps[1][0, 0, NBOX + 2] = 3.0
        pred = yolov8_raw_to_pred(maps, 640)
        np.testing.assert_allclose(pred[0, 4:], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(pred[6400, 4:], [0.5, 0.5, 1 / (1 + np.exp(-3))], rtol=1e-6)

    def test_split_box_and_class_maps_match_combined(self):
        rng = np.random.default_rng(0)
        maps = [rng.normal(size=(640 // s, 640 // s, NBOX + 5)).astype(np.float32)
                for s in (8, 16, 32)]
        split = [part for m in maps for part in (m[..., NBOX:], m[..., :NBOX])]
        rng.shuffle(split)
        np.testing.assert_array_equal(yolov8_raw_to_pred(split, 640),
                                      yolov8_raw_to_pred(maps, 640))


# ---------------------------------------------------------------------------
# DpuDetector with a fake VART runner
# ---------------------------------------------------------------------------

class FakeTensor:
    def __init__(self, dims, dtype="xint8", fix_point=0):
        self.dims, self.dtype, self._fp = list(dims), dtype, fix_point

    def get_attr(self, name):
        assert name == "fix_point"
        return self._fp


class FakeRunner:
    """VART runner interface; `produce(input_array) -> list of output arrays`."""

    def __init__(self, inputs, outputs, produce):
        self._in, self._out, self.produce = inputs, outputs, produce
        self.last_input = None

    def get_input_tensors(self):
        return self._in

    def get_output_tensors(self):
        return self._out

    def execute_async(self, inputs, outputs):
        self.last_input = inputs[0].copy()
        for buf, value in zip(outputs, self.produce(inputs[0])):
            buf[...] = value
        return (1, 0)

    def wait(self, job):
        assert job == (1, 0)


class TestDpuDetectorFakeRunner(unittest.TestCase):
    IN_FP, OUT_FP = 7, 2  # input real = q / 128, output real = q / 4

    def int8_maps(self):
        """Quantized raw head with one strong 'person' at stride-16 cell (y=20, x=10)."""
        nc = 80
        maps = []
        for s in (8, 16, 32):
            g = 640 // s
            m = np.full((1, g, g, NBOX + nc), -128, np.int8)   # real -32: all scores ~0
            m[..., :NBOX] = 0
            maps.append(m)
        cell = maps[1][0, 20, 10]
        cell[:NBOX] = np.where(peaked_dfl(3, 4, 3, 4, peak=1) > 0, 127, -128)  # l t r b bins
        cell[NBOX + 0] = 127                                                    # person
        return maps

    def make(self, in_dtype="xint8", out_dtype="xint8", maps=None):
        maps = maps if maps is not None else self.int8_maps()
        runner = FakeRunner(
            [FakeTensor((1, 640, 640, 3), in_dtype, self.IN_FP)],
            [FakeTensor(m.shape, out_dtype, self.OUT_FP) for m in maps],
            lambda x: maps,
        )
        return DpuDetector(runner=runner), runner

    def test_quantized_end_to_end(self):
        det, runner = self.make()
        frame = np.full((576, 768, 3), 255, np.uint8)
        d = det.detect(frame)

        # input: NHWC int8, white -> 1.0 * 128 clipped to 127, pad 114/255*128 -> 57
        x = runner.last_input
        self.assertEqual((x.shape, x.dtype), ((1, 640, 640, 3), np.int8))
        self.assertEqual((x[0, 320, 320, 0], x[0, 0, 0, 0]), (127, 57))

        # anchor (10.5, 20.5), distances l3 t4 r3 b4 -> letterbox (120, 264)-(216, 392)
        # -> frame: x / gain, (y - 80) / gain with gain 640/768
        self.assertEqual(len(d), 1)
        self.assertEqual(d.labels, ["person"])
        np.testing.assert_allclose(d.boxes[0], [144, 220.8, 259.2, 374.4], atol=0.05)
        self.assertGreater(d.scores[0], 0.99)
        self.assertEqual(set(det.timings), {"pre", "inference", "post"})

    def test_float_tensors_are_not_scaled(self):
        maps = [m.astype(np.float32) / 4 for m in self.int8_maps()]
        det, runner = self.make("float32", "float32", maps)
        d = det.detect(np.full((576, 768, 3), 255, np.uint8))
        self.assertAlmostEqual(float(runner.last_input[0, 320, 320, 0]), 1.0)
        np.testing.assert_allclose(d.boxes[0], [144, 220.8, 259.2, 374.4], atol=0.05)

    def test_rejects_nchw_input(self):
        runner = FakeRunner([FakeTensor((1, 3, 640, 640))], [], lambda x: [])
        with self.assertRaises(ValueError):
            DpuDetector(runner=runner)


# ---------------------------------------------------------------------------
# Cross-check against the real YOLOv8n (dev machine only)
# ---------------------------------------------------------------------------

def _have_reference():
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return os.path.exists(os.path.join(ROOT, "yolov8n.pt"))


@unittest.skipUnless(_have_reference(), "needs torch, ultralytics and yolov8n.pt")
class TestAgainstYolov8n(unittest.TestCase):
    """Raw Detect-head maps from the real model, decoded by our code, must match
    the model's own decoded output, and DpuDetector (fed those maps through a
    fake runner) must produce the same detections as decoding that output."""

    @classmethod
    def setUpClass(cls):
        import cv2
        import torch
        from ultralytics import YOLO
        from ultralytics.utils import ASSETS

        cls.torch = torch
        cls.model = YOLO(os.path.join(ROOT, "yolov8n.pt")).model.float().eval()
        cls.head = cls.model.model[-1]
        cls.frame = cv2.imread(str(ASSETS / "bus.jpg"))

    def run_model(self, x_nhwc):
        """(full decoded output (84, 8400), raw head maps as NHWC arrays)."""
        torch, head = self.torch, self.head
        feats = {}
        hook = head.register_forward_pre_hook(
            lambda mod, args: feats.__setitem__("x", [t.clone() for t in args[0]]))
        try:
            with torch.no_grad():
                x = torch.from_numpy(np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2)))
                y = self.model(x)
                y = y[0] if isinstance(y, (tuple, list)) else y
                raw = [torch.cat([head.cv2[i](f), head.cv3[i](f)], 1)
                       for i, f in enumerate(feats["x"])]
        finally:
            hook.remove()
        return y[0].numpy(), [r[0].permute(1, 2, 0).numpy() for r in raw]

    def test_head_decoding_matches_model(self):
        from detector import rgb_unit

        img, _, _ = letterbox(self.frame, 640)
        y, raw = self.run_model(rgb_unit(img)[None])
        pred = yolov8_raw_to_pred(raw, 640)
        self.assertEqual(pred.shape, y.T.shape)
        np.testing.assert_allclose(pred[:, :4], y.T[:, :4], atol=2e-3)
        np.testing.assert_allclose(pred[:, 4:], y.T[:, 4:], atol=1e-5)

    def test_dpu_detector_matches_model_output(self):
        holder = {}

        def produce(x):
            holder["y"], raw = self.run_model(x)
            return [r[None] for r in raw]

        shapes = [(1, 640 // s, 640 // s, NBOX + 80) for s in (8, 16, 32)]
        runner = FakeRunner([FakeTensor((1, 640, 640, 3), "float32")],
                            [FakeTensor(s, "float32") for s in shapes], produce)
        det = DpuDetector(runner=runner)
        ours = det.detect(self.frame)

        _, gain, pad = letterbox(self.frame, 640)
        ref = decode_predictions(holder["y"].T, 0.1, 0.7, gain, pad,
                                 self.frame.shape[:2], det.names)
        self.assertGreater(len(ref), 0)
        self.assertEqual(ours.class_ids.tolist(), ref.class_ids.tolist())
        np.testing.assert_allclose(ours.boxes, ref.boxes, atol=0.05)
        np.testing.assert_allclose(ours.scores, ref.scores, atol=1e-5)
        self.assertIn("person", ours.labels)


if __name__ == "__main__":
    unittest.main()
