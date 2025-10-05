import unittest
from src.qlora_finetune_base.evaluation.evaluator import Evaluator
from src.qlora_finetune_base.evaluation.metrics import compute_metrics

class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.evaluator = Evaluator()
        self.sample_predictions = [0.9, 0.1, 0.8, 0.4]
        self.sample_labels = [1, 0, 1, 0]

    def test_compute_metrics(self):
        metrics = compute_metrics(self.sample_predictions, self.sample_labels)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)

    def test_evaluator_initialization(self):
        self.assertIsNotNone(self.evaluator)

    def test_evaluate(self):
        results = self.evaluator.evaluate(self.sample_predictions, self.sample_labels)
        self.assertIsInstance(results, dict)
        self.assertIn('accuracy', results)

if __name__ == '__main__':
    unittest.main()