from test.unit.test_helpers import TensorTestCase

from joeynmt.metrics import token_accuracy, chrf


class TestMetrics(TensorTestCase):
    def test_token_acc_level_char(self):
        hyp = ["test"]
        ref = ["tezt"]
        level = "char"
        acc = token_accuracy(hyp, ref, level)
        self.assertEqual(acc, 75)

    def test_chrf(self):
        hyp = ["tes"]
        ref = ["test"]
        metric = chrf(hyp, ref)
        self.assertIsInstance(metric, float)
        self.assertEqual(round(metric), 69)
