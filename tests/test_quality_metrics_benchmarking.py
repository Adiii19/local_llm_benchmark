import unittest

from src.benchmarking.benchmarking_suite import BenchmarkSuite


class BenchmarkQualityMetricsTest(unittest.TestCase):
    def test_evaluate_generated_outputs_uses_quality_metrics(self):
        suite = BenchmarkSuite.__new__(BenchmarkSuite)

        generated = [
            "Machine learning is a subset of artificial intelligence",
            "Neural networks use layers of connected nodes",
        ]
        references = [
            "Machine learning is a subset of AI that learns from data",
            "Neural networks are layers of connected neurons",
        ]

        metrics = suite.evaluate_generated_outputs(generated, references)

        self.assertIn("mean_bleu", metrics)
        self.assertIn("mean_rouge_l", metrics)
        self.assertIn("mean_similarity", metrics)
        self.assertEqual(metrics["num_pairs"], 2)
        self.assertGreaterEqual(metrics["mean_similarity"], 0.0)


if __name__ == "__main__":
    unittest.main()
