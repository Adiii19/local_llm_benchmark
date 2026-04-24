"""
src/evaluation/quality_metrics.py

Measure answer quality
BLEU, ROUGE, semantic similarity, etc.
"""

from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
import math


class QualityMetrics:
    """Calculate quality metrics for generated text"""
    
    @staticmethod
    def bleu_score(reference: str, hypothesis: str, n_grams: int = 4) -> float:
        """
        BLEU Score: 0-1, higher is better
        Measures n-gram overlap with reference
        
        Common in MT evaluation
        """
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if not hyp_tokens:
            return 0.0
        
        # Calculate n-gram matches
        score = 1.0
        
        for n in range(1, min(n_grams + 1, len(ref_tokens) + 1)):
            ref_ngrams = Counter(
                tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)
            )
            hyp_ngrams = Counter(
                tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1)
            )
            
            matches = sum((hyp_ngrams & ref_ngrams).values())
            max_matches = max(sum(hyp_ngrams.values()), 1)
            
            score *= matches / max_matches if max_matches > 0 else 0
        
        return min(score, 1.0)
    
    @staticmethod
    def rouge_l(reference: str, hypothesis: str) -> float:
        """
        ROUGE-L: Longest common subsequence metric
        Good for summary evaluation
        """
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        def lcs_length(a: List, b: List) -> int:
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs = lcs_length(ref_tokens, hyp_tokens)
        
        if not ref_tokens or not hyp_tokens:
            return 0.0
        
        # F-score
        precision = lcs / len(hyp_tokens) if hyp_tokens else 0
        recall = lcs / len(ref_tokens) if ref_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f_score = 2 * (precision * recall) / (precision + recall)
        return f_score
    
    @staticmethod
    def semantic_similarity(reference: str, hypothesis: str) -> float:
        """
        Simple word overlap similarity
        (Advanced: use embeddings from SentenceTransformer)
        """
        ref_tokens = set(reference.lower().split())
        hyp_tokens = set(hypothesis.lower().split())
        
        if not ref_tokens or not hyp_tokens:
            return 0.0
        
        intersection = ref_tokens & hyp_tokens
        union = ref_tokens | hyp_tokens
        
        # Jaccard similarity
        return len(intersection) / len(union)
    
    @staticmethod
    def answer_length_ratio(reference: str, hypothesis: str) -> float:
        """
        Check if generated answer is reasonable length
        Too short (incomplete) or too long (verbose)
        """
        ref_len = len(reference.split())
        hyp_len = len(hypothesis.split())
        
        if ref_len == 0:
            return 1.0 if hyp_len == 0 else 0.5
        
        ratio = hyp_len / ref_len
        
        # Penalty if too different
        if ratio < 0.5:
            return ratio  # Too short
        elif ratio > 2.0:
            return 2.0 / ratio  # Too long
        else:
            return 1.0  # Just right
    
    @staticmethod
    def evaluate_multiple(
        references: List[str],
        hypotheses: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate multiple pairs and aggregate
        
        Args:
            references: Expected answers
            hypotheses: Generated answers
        
        Returns:
            Aggregated metrics
        """
        assert len(references) == len(hypotheses)
        
        bleu_scores = []
        rouge_scores = []
        sim_scores = []
        length_ratios = []
        
        for ref, hyp in zip(references, hypotheses):
            bleu_scores.append(QualityMetrics.bleu_score(ref, hyp))
            rouge_scores.append(QualityMetrics.rouge_l(ref, hyp))
            sim_scores.append(QualityMetrics.semantic_similarity(ref, hyp))
            length_ratios.append(QualityMetrics.answer_length_ratio(ref, hyp))
        
        return {
            'mean_bleu': np.mean(bleu_scores),
            'mean_rouge_l': np.mean(rouge_scores),
            'mean_similarity': np.mean(sim_scores),
            'mean_length_ratio': np.mean(length_ratios),
            'num_pairs': len(references)
        }


if __name__ == "__main__":
    # Test examples
    ref = "Machine learning is a subset of artificial intelligence"
    hyp = "Machine learning is part of AI"
    
    print(f"Reference: {ref}")
    print(f"Hypothesis: {hyp}\n")
    print(f"BLEU: {QualityMetrics.bleu_score(ref, hyp):.3f}")
    print(f"ROUGE-L: {QualityMetrics.rouge_l(ref, hyp):.3f}")
    print(f"Similarity: {QualityMetrics.semantic_similarity(ref, hyp):.3f}")