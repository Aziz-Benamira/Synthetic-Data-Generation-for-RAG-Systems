import ragas 
import pytrec_eval
# 1- Retriever evaluation metrics 
# 1.1- with ground truth
def retrieverEvaluator(ground_truth, metrics):
    """ Evalutor for the retriever part of a RAG, when a ground-truth is known.
    """
    return pytrec_eval.RelevanceEvaluator(ground_truth, metrics)