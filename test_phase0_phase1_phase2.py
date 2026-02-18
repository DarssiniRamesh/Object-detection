import unittest
import math
import random
from types import SimpleNamespace

import torch

from util.misc import nested_tensor_from_tensor_list, NestedTensor, SmoothedValue, MetricLogger
from util import box_ops
from datasets import transforms as det_transforms
from models.matcher import HungarianMatcher
from models.detr import SetCriterion, PostProcess
from models.segmentation import dice_loss, sigmoid_focal_loss
from models.transformer import Transformer


class TestPhase0Utilities(unittest.TestCase):
    """Phase 0: fast, deterministic unit tests for core utilities."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        random.seed(0)

    def test_nested_tensor_from_tensor_list_padding_and_mask(self):
        """nested_tensor_from_tensor_list should pad to max H/W and mark padded pixels True in mask."""
        img1 = torch.zeros(3, 10, 12)
        img2 = torch.zeros(3, 8, 9)
        nt = nested_tensor_from_tensor_list([img1, img2])

        self.assertIsInstance(nt, NestedTensor)
        self.assertEqual(tuple(nt.tensors.shape), (2, 3, 10, 12))
        self.assertEqual(tuple(nt.mask.shape), (2, 10, 12))

        # For first image, no padding in its own extent: mask should be False everywhere.
        self.assertTrue((nt.mask[0] == False).all().item())

        # For second image: pixels beyond (8,9) should be padded => mask True.
        self.assertTrue((nt.mask[1, :8, :9] == False).all().item())
        self.assertTrue((nt.mask[1, 8:, :] == True).all().item())
        self.assertTrue((nt.mask[1, :, 9:] == True).all().item())

    def test_nested_tensor_to_moves_mask_and_tensor(self):
        """NestedTensor.to should move both tensors and mask to the given device."""
        nt = nested_tensor_from_tensor_list([torch.rand(3, 5, 6)])
        nt2 = nt.to(torch.device("cpu"))
        self.assertEqual(nt2.tensors.device.type, "cpu")
        self.assertEqual(nt2.mask.device.type, "cpu")

    def test_smoothed_value_basic_stats(self):
        sv = SmoothedValue(window_size=3)
        sv.update(1.0)
        sv.update(2.0)
        sv.update(3.0)
        self.assertAlmostEqual(sv.median, 2.0, places=6)
        self.assertAlmostEqual(sv.avg, 2.0, places=6)
        self.assertAlmostEqual(sv.global_avg, 2.0, places=6)
        self.assertEqual(sv.value, 3.0)

        # window should slide
        sv.update(100.0)
        self.assertAlmostEqual(sv.median, 3.0, places=6)
        self.assertAlmostEqual(sv.avg, (2.0 + 3.0 + 100.0) / 3.0, places=6)

    def test_metric_logger_update_tensor_and_numeric(self):
        ml = MetricLogger()
        ml.update(loss=torch.tensor(3.0), lr=1e-3)
        self.assertIn("loss", ml.meters)
        self.assertIn("lr", ml.meters)
        self.assertAlmostEqual(ml.meters["loss"].value, 3.0, places=6)
        self.assertAlmostEqual(ml.meters["lr"].value, 1e-3, places=12)

    def test_box_ops_generalized_iou_identity_is_one(self):
        """GIoU for identical boxes should be 1 (on diagonal)."""
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [2.0, 3.0, 8.0, 9.0]])
        giou = box_ops.generalized_box_iou(boxes, boxes)
        self.assertTrue(torch.allclose(torch.diag(giou), torch.ones(2), atol=1e-6))

    def test_box_ops_masks_to_boxes_empty(self):
        masks = torch.zeros((0, 5, 5), dtype=torch.uint8)
        boxes = box_ops.masks_to_boxes(masks)
        self.assertEqual(tuple(boxes.shape), (0, 4))


class TestPhase1Transforms(unittest.TestCase):
    """Phase 1: test data transforms behavior for boxes/masks/size bookkeeping."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        random.seed(0)

    def _make_pil_image(self, w=20, h=10):
        # Use torchvision functional to_tensor expects PIL; simplest via PIL.Image
        from PIL import Image

        # solid gray image
        return Image.fromarray((torch.ones(h, w, 3) * 127).byte().numpy())

    def test_hflip_flips_boxes(self):
        img = self._make_pil_image(w=20, h=10)
        target = {
            "boxes": torch.tensor([[2.0, 1.0, 6.0, 5.0]]),  # xyxy
        }
        img2, t2 = det_transforms.hflip(img, target)
        self.assertEqual(img2.size, img.size)

        # For width=20: new_x0 = w - old_x1, new_x1 = w - old_x0
        expected = torch.tensor([[20.0 - 6.0, 1.0, 20.0 - 2.0, 5.0]])
        self.assertTrue(torch.allclose(t2["boxes"], expected))

    def test_resize_updates_boxes_area_and_size(self):
        img = self._make_pil_image(w=20, h=10)
        target = {
            "boxes": torch.tensor([[2.0, 1.0, 6.0, 5.0]]),
            "area": torch.tensor([16.0]),
        }
        # Resize min_size to 20 => since w=20,h=10 and h<=w => new_h=20, new_w=40
        img2, t2 = det_transforms.resize(img, target, size=20, max_size=None)
        self.assertEqual(img2.size, (40, 20))  # PIL size = (w,h)

        ratio_w = 40.0 / 20.0
        ratio_h = 20.0 / 10.0
        expected_boxes = target["boxes"] * torch.tensor([ratio_w, ratio_h, ratio_w, ratio_h])
        self.assertTrue(torch.allclose(t2["boxes"], expected_boxes, atol=1e-6))
        self.assertTrue(torch.allclose(t2["area"], target["area"] * (ratio_w * ratio_h), atol=1e-6))
        self.assertTrue(torch.equal(t2["size"], torch.tensor([20, 40])))

    def test_pad_updates_size_and_masks(self):
        img = self._make_pil_image(w=5, h=4)
        target = {"size": torch.tensor([4, 5]), "masks": torch.zeros((1, 4, 5), dtype=torch.uint8)}
        img2, t2 = det_transforms.pad(img, target, padding=(3, 2))  # pad right=3, bottom=2
        self.assertEqual(img2.size, (8, 6))
        self.assertTrue(torch.equal(t2["size"], torch.tensor([6, 8])))
        self.assertEqual(tuple(t2["masks"].shape), (1, 6, 8))

    def test_normalize_converts_boxes_to_cxcywh_and_normalizes(self):
        # Use ToTensor first to get tensor image for Normalize
        img = self._make_pil_image(w=20, h=10)
        img_t, target = det_transforms.ToTensor()(img, {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]])})
        norm = det_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
        img_n, t2 = norm(img_t, target)

        self.assertEqual(tuple(img_n.shape[-2:]), (10, 20))
        # xyxy -> cxcywh => (5,5,10,10) then / (w,h,w,h) => (0.25,0.5,0.5,1.0)
        expected = torch.tensor([[0.25, 0.5, 0.5, 1.0]])
        self.assertTrue(torch.allclose(t2["boxes"], expected, atol=1e-6))


class TestPhase2ModelLogic(unittest.TestCase):
    """Phase 2: deeper logic tests for criterion losses, post-processing, transformer shapes."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        random.seed(0)

    def test_setcriterion_loss_labels_includes_class_error(self):
        num_classes = 3
        matcher = HungarianMatcher(cost_class=1, cost_bbox=0, cost_giou=0)
        criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict={"loss_ce": 1.0},
            eos_coef=0.1,
            losses=["labels"],
        )

        # batch=1, queries=2, classes+1=4
        # Query 0 should clearly predict class 1, Query 1 predicts "no-object" (class index 3)
        logits = torch.tensor([[[0.0, 10.0, 0.0, -10.0], [0.0, 0.0, 0.0, 10.0]]])
        boxes = torch.rand(1, 2, 4)
        outputs = {"pred_logits": logits, "pred_boxes": boxes}
        targets = [{"labels": torch.tensor([1]), "boxes": torch.rand(1, 4)}]

        losses = criterion(outputs, targets)
        self.assertIn("loss_ce", losses)
        self.assertIn("class_error", losses)
        self.assertTrue(torch.isfinite(losses["loss_ce"]).item())
        self.assertGreaterEqual(losses["class_error"].item(), 0.0)
        self.assertLessEqual(losses["class_error"].item(), 100.0)

    def test_setcriterion_loss_boxes_zero_when_exact_match(self):
        num_classes = 2
        matcher = HungarianMatcher(cost_class=0, cost_bbox=1, cost_giou=0)
        criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict={"loss_bbox": 1.0, "loss_giou": 1.0},
            eos_coef=0.1,
            losses=["boxes"],
        )

        # Make predicted boxes equal targets => loss_bbox should be 0 and loss_giou should be 0.
        pred_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.4]]], dtype=torch.float32)
        logits = torch.zeros(1, 1, num_classes + 1)
        outputs = {"pred_logits": logits, "pred_boxes": pred_boxes}
        targets = [{"labels": torch.tensor([0]), "boxes": pred_boxes[0].clone()}]

        losses = criterion(outputs, targets)
        self.assertIn("loss_bbox", losses)
        self.assertIn("loss_giou", losses)
        self.assertAlmostEqual(losses["loss_bbox"].item(), 0.0, places=6)
        self.assertAlmostEqual(losses["loss_giou"].item(), 0.0, places=6)

    def test_postprocess_scales_boxes_to_absolute_coords(self):
        pp = PostProcess()

        # pred box is cxcywh normalized, for a 10x20 (h,w) image.
        out_bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.4]]])  # => xyxy = [0.4,0.3,0.6,0.7]
        logits = torch.tensor([[[0.0, 10.0]]])  # 1 class + no-object
        outputs = {"pred_logits": logits, "pred_boxes": out_bbox}
        target_sizes = torch.tensor([[10, 20]])  # h,w

        results = pp(outputs, target_sizes)
        self.assertEqual(len(results), 1)
        self.assertIn("boxes", results[0])
        abs_box = results[0]["boxes"][0]
        expected = torch.tensor([0.4 * 20, 0.3 * 10, 0.6 * 20, 0.7 * 10])
        self.assertTrue(torch.allclose(abs_box, expected, atol=1e-6))

    def test_dice_loss_bounds(self):
        inputs = torch.zeros((2, 16), dtype=torch.float32)
        targets = torch.zeros((2, 16), dtype=torch.float32)
        loss = dice_loss(inputs, targets, num_boxes=2.0)
        # Should be finite and in [0, 1] for this trivial case.
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertLessEqual(loss.item(), 1.0)

    def test_sigmoid_focal_loss_zero_for_easy_correct(self):
        # Large positive logits for positive targets and large negative logits for negative targets
        inputs = torch.tensor([[10.0, -10.0]], dtype=torch.float32)
        targets = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        loss = sigmoid_focal_loss(inputs, targets, num_boxes=1.0)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertLess(loss.item(), 1e-3)

    def test_transformer_output_shapes(self):
        # Small transformer for fast CPU test
        tr = Transformer(
            d_model=32,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=64,
            dropout=0.0,
            return_intermediate_dec=True,
        )

        bs, c, h, w = 2, 32, 4, 5
        src = torch.randn(bs, c, h, w)
        mask = torch.zeros(bs, h, w, dtype=torch.bool)
        query_embed = torch.randn(10, 32)  # num_queries=10
        pos_embed = torch.randn(bs, c, h, w)

        hs, memory = tr(src, mask, query_embed, pos_embed)
        # hs: [num_layers, bs, num_queries, d_model]
        self.assertEqual(tuple(hs.shape), (2, 2, 10, 32))
        # memory: [bs, c, h, w]
        self.assertEqual(tuple(memory.shape), (2, 32, 4, 5))


if __name__ == "__main__":
    unittest.main()
