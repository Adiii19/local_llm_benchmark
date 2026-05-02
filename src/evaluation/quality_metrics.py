from typing import List,Dict, Tuple
import numpy as np
from collections import Counter
import math

class QualityMetrics:

    @staticmethod
    def bleu_score(reference:str,hypothesis:str,n_grams:int=4)->float:
        
        ref_tokens=reference.lower().split()
        hyp_tokens=hypothesis.lower().split()

        if not hyp_tokens:
            return 0.0
        
      # Calculate n-gram matches
        score=1.0

        for n in range(1,min(n_grams+1,len(ref_tokens)+1)):
            ref_ngrams=Counter(
                tuple(ref_tokens[i:i+n] for i in range(len(ref_tokens)-n+1))

            )
            hyp_ngrams=Counter(
                tuple(hyp_tokens[i:i+n]for i in range(len(hyp_tokens)-n+1))
            )

            mathces=sum((hyp_ngrams&ref_ngrams).values())
            max_matches=max(sum(hyp_ngrams.values()),1)

            score*=mathces/max_matches if max_matches>0 else 0

        return min(score,1.0)
    
    @staticmethod
    def rouge_l(refernce:str,hypothesis:str)->float:

        ref_tokens=refernce.lower().split()
        hyp_tokens=hypothesis.lower().split()

        def lcs_length(a:List,b:List)->int:
            m,n=len(a),len(b)
            dp=[[0]*(n+1) for _ in range(m+1)]

            for i in range(1,m+1):
                for j in range(1,n+1):
                    if a[i-1]==b[j-1]:
                        dp[i][j]=dp[i-1][j-1]+1
                    else:
                        dp[i][j]=max(dp[i-1][j],dp[i][j-1])

            return dp[m][n]
        
        lcs=lcs_length(ref_tokens,hyp_tokens)

        if not ref_tokens or not hyp_tokens:
            return 0.0
        
        precision=lcs/len(hyp_tokens) if hyp_tokens else 0
        recall=lcs/len(ref_tokens) if ref_tokens else 0

        if precision + recall==0:
            return 0.0
        f_score=2*(precision*recall)/(precision+recall)
        return f_score
    
    @staticmethod
    def semantic_similarity(reference:str,hypothesis:str)->float:

        ref_tokens=set(reference.lower().split())
        hyp_tokens=set(hypothesis.lower().split())

        if not ref_tokens or not hyp_tokens:
            return 0.0
        
        intersection=ref_tokens&hyp_tokens
        union=ref_tokens|hyp_tokens

        return len(intersection)/len(union)
    
    @staticmethod
    def answer_length_ratio(references:str,hypothesis:str)->float:
        ref_len=len(references.split())
        hyp_len=len(hypothesis.split())

        if ref_len==0:
            return 1.0 if hyp_len==0 else 0.5
        
        ratio=hyp_len/ref_len

        if ratio<0.5:
            return ratio
        elif ratio>2.0:
            return 2.0/ratio
        else:
            return 1.0
        

    @staticmethod
    def evaluate_multiple(
        references:List[str],
        hypothesis:List[str]
    )->Dict[str,float]:
        
        assert len(references)==len(hypothesis)

        bleu_scores=[]
        rouge_Scores=[]
        sim_scores=[]
        length_ratios=[]

        for ref,hyp in zip(references,hypothesis):
            bleu_scores.append(QualityMetrics.bleu_score(ref,hyp))
            rouge_Scores.append(QualityMetrics.rouge_l(ref,hyp))
            sim_scores.append(QualityMetrics.semantic_similarity(ref,hyp))
            length_ratios.append(QualityMetrics.answer_length_ratio(ref,hyp))

        return {

            'mean_bleu':np.mean(bleu_scores),
            'mean_rouge_l':np.mean(rouge_Scores),
            'mean_similarity': np.mean(sim_scores),
            'mean_length_ratio': np.mean(length_ratios),
            'num_pairs': len(references)



        }

if __name__=="__main__":
    ref="Machine learning is a subset of artificial intelligence"
    hyp="Machine learning is a part of AI"

    print(f"Reference: {ref}")
    print(f"Hypothesis: {hyp}\n")
    print(f"BLEU: {QualityMetrics.bleu_score(ref, hyp):.3f}")
    print(f"ROUGE-L: {QualityMetrics.rouge_l(ref, hyp):.3f}")
    print(f"Similarity: {QualityMetrics.semantic_similarity(ref, hyp):.3f}")

        
        