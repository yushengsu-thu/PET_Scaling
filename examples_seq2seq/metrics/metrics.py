# several of the evaluation metrics are from https://github.com/google-research/text-to-text-transfer-transformer/blob/a1352e625db7ec114062f99d99b0565b9e45c155/t5/evaluation/metrics.py
"""Defines different metrics used for evaluation of tasks."""
import numpy as np
import scipy
import math
import sklearn
import collections
from logging import getLogger
from .qa_utils import normalize_squad, qa_metrics
import sklearn.metrics
import rouge
import nltk
import re
from collections import Counter
import string

logger = getLogger(__name__)



def accuracy(predictions, targets) -> dict:
    """Computes the average accuracy."""
    return {"accuracy": 100 * ((np.array(predictions) == np.array(targets)).mean())}

def pearson_corrcoef(predictions, targets) -> dict:
    """Computes Pearson correlation coefficient."""
    from examples_seq2seq.data_processors.postprocessors import string_to_float
    targets = [string_to_float(target) for target in targets]
    predictions= [string_to_float(prediction) for prediction in predictions]
    pearson_corrcoef = 100 * scipy.stats.pearsonr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(pearson_corrcoef):
        pearson_corrcoef = 0
    return {"pearson": pearson_corrcoef}


def spearman_corrcoef(predictions, targets) -> dict:
    """Computes Spearman correlation coefficient."""
    # TODO: we need to do postprocessors in a clean way for each dataset.
    from examples_seq2seq.data_processors.postprocessors import string_to_float
    targets = [string_to_float(target) for target in targets]
    predictions= [string_to_float(prediction) for prediction in predictions]
    spearman_corrcoef = 100 * scipy.stats.spearmanr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(spearman_corrcoef):
        spearman_corrcoef = 0
    return {"spearmanr": spearman_corrcoef}


def f1_score_with_invalid(predictions, targets) -> dict:
    """Computes F1 score,  with any prediction != 0 or 1 is counted as incorrect.
    Args:
      targets: list of targets, either 0 or 1
      predictions: list of predictions, any integer value
    Returns:
      F1 score, where any prediction != 0 or 1 is counted as wrong.
    """
    def binary_reverse(labels):
       return ['0' if label == '1' else '1' for label in labels]
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    # Get indices of invalid predictions.
    invalid_idx_mask = np.logical_and(predictions != '0', predictions != '1')
    # For any prediction != 0 or 1, we set the prediction to the opposite of its corresponding target.
    predictions[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask])
    targets = targets.astype(np.int32)
    predictions = predictions.astype(np.int32)
    return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions)}

# TODO: maybe gaurd against invalid values https://stackoverflow.com/questions/56865344/how-do-i-calculate-the-matthews-correlation-coefficient-in-tensorflow
def matthews_corrcoef(predictions, targets) -> dict:
    """Computes the Matthews correlation coefficient."""
    return {"matthews_correlation": 100 * sklearn.metrics.matthews_corrcoef(targets, predictions)}

def squad(predictions, targets):
  """Computes SQuAD metrics, maximizing over answers per question.
  Args:
    targets: list of lists of strings
    predictions: list of strings
  Returns:
    dict with score_key: squad score across all targets and predictions
  """

  targets = [[normalize_squad(t) for t in u] for u in targets]
  predictions = [normalize_squad(p) for p in predictions]
  return qa_metrics(targets, predictions)

def my_squad(predictions, targets):
    acc_result = {"train": False, "total": 0, "em_sum": 0, "f1_sum": 0., "NA_tp": 0, "NA_fp": 0, "NA_tn": 0,
                  "NA_fn": 0}
    pred = []
    ground = []
    for p in predictions:
        pred.append(normalize_answer(p))
    # pred = [normalize_answer([int(n) for n in p if n == 1 break], skip_special_tokens=True)) for p in predict]
    for ans in targets:
        ground.append({normalize_answer(a) for a in ans})

    # print(pred)
    # print(ground)
    # print("==" * 10)
    acc_result["em_sum"] += squad_em(pred, ground)
    acc_result["f1_sum"] += squad_f1(pred, ground)
    acc_result["total"] += len(pred)
    acc_result = squad_NAF1(pred, ground, acc_result)
    # print(acc_result)
    # print('rank here:{}'.format(os.environ['LOCAL_RANK']))
    return acc_result



def exact_match(predictions, targets):
  """Computes whether the targets match predictions exactly."""
  return {"em": 100 * float(np.array_equal(targets, predictions))}


def sklearn_metrics_wrapper(metric_str,
                            metric_dict_str=None,
                            metric_post_process_fn=None,
                            **metric_fn_kwargs):
  """Wraps any sklearn.metric function and returns a t5 metric function.
  Args:
    metric_str: string, the function from `sklearn.metrics` to use.
    metric_dict_str: optional string, if not specified `metric_str` is used as
      the key in the returned dictionary.
    metric_post_process_fn: callable, if specified the final computed metric
      will be passed through this.
    **metric_fn_kwargs: kwargs, passed to the metric function we are calling.
  Returns:
    the function that calculates the metric in a dict.
  """
  if not hasattr(sklearn.metrics, metric_str):
    raise ValueError("sklearn.metrics does not have: %s" % metric_str)

  def fn(predictions, targets):
    metric_fn = getattr(sklearn.metrics, metric_str)
    metric_val = metric_fn(targets, predictions, **metric_fn_kwargs)
    if metric_post_process_fn is not None:
      metric_val = metric_post_process_fn(metric_val)
    return {metric_dict_str or metric_str: metric_val}
  return fn


def mean_multiclass_f1(num_classes, **metric_fn_kwargs):
  """Computes the unweighted average of the F1 per class."""
  return sklearn_metrics_wrapper(
      "fbeta_score",
      metric_dict_str="f1_multiclass",
      metric_post_process_fn=lambda x: 100 * x,
      beta=1,
      labels=range(num_classes),
      average="macro",
      **metric_fn_kwargs)

def mean_multiclass_f1_mathqa(num_classes, **metric_fn_kwargs):
  """Computes the unweighted average of the F1 per class."""
  return sklearn_metrics_wrapper(
      "fbeta_score",
      metric_dict_str="f1_multiclass_mathqa",
      metric_post_process_fn=lambda x: 100 * x,
      beta=1,
      labels=['a', 'b', 'c', 'd', 'e'],
      average="macro",
      **metric_fn_kwargs)

def rouge_l(predictions, targets):
  """ROUGE-L score."""
  
  evaluator = rouge.Rouge(metrics=['rouge-l'], max_n=4, weight_factor=1.2, stemming=True)
  return {'rouge_l': 100 * evaluator.get_scores(predictions, targets)['rouge-l']['f']}

def multirc_f1_over_all_answers(targets, predictions):
  """Special metric for MultiRC which computes F1 score over all examples.
  This is necessary because the targets/predictions for MultiRC are dicts and
  the f1_score_with_invalid expects a list of True/False labels, not dicts. As
  a result we just need to key in the "value" for each of the example dicts
  before feeding into f1_score_with_invalid.
  Args:
    targets: list of dicts, where each dict has a "value" key.
    predictions: list of dicts, where each dict has a "value" key.
  Returns:
    F1 score over values, where any prediction != 0 or 1 is counted as wrong.
  """
  return f1_score_with_invalid(
      [t["value"] for t in targets], [p["value"] for p in predictions]
  )


def mean_group_metric(metric_fn, group_key="group", value_key="value"):
  """Returns a metric that averages `metric_fn` on sub-groups of results.
  The sub-groups are defined by aggregating results (targets and predictions)
  by accessing the feature specified by `group_key` in the target dicts.
  **WARNING**: Using this function can produce unreliable results if you do not
  pass in full groups. For example, if you evaluate over a random subsample of a
  validation set and do not retain all of the examples in each group, you may
  get results which aren't directly comparable to using the full validation set.
  Args:
    metric_fn: function, the metric to compute on the subgroups.
    group_key: string, the key for the grouping value in the target dictionary.
    value_key: string, the key for the value in the dictionaries.
  """
  def my_metric(targets, predictions):
    """Computes mean of `metric_fn` over subgroups of results."""
    grouped_values = collections.defaultdict(lambda: ([], []))
    for targ, pred in zip(targets, predictions):
      g = targ[group_key]
      grouped_values[g][0].append(targ[value_key])
      grouped_values[g][1].append(pred[value_key])
    group_scores = collections.defaultdict(list)
    for (targets, predictions) in grouped_values.values():
      for metric, score in metric_fn(targets, predictions).items():
        group_scores[metric].append(score)
    return {metric: np.mean(scores) for metric, scores in group_scores.items()}
  return my_metric


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def squad_em(predict, answers):
    em = 0
    for pre, ans in zip(predict, answers):
        if pre in ans:
            em += 1
    return em

def squad_f1(predict, answers):
    ret = 0
    for pred, ans in zip(predict, answers):
        # if pred == "no answer":
        #     continue
        prediction_tokens = pred.split()
        cpred_token = Counter(prediction_tokens)
        curf1 = []
        for a in ans:
            ground_truth_tokens = a.split()
            common = cpred_token & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0 or len(prediction_tokens) == 0:
                curf1.append(0)
            else:
                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(ground_truth_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
                curf1.append(f1)
        ret += max(curf1)
    return ret

def squad_NAF1(predict, answers, acc_result):
    for p, ans in zip(predict, answers):
        if p == "no answer":
            if "no answer" in ans:
                acc_result["NA_tp"] += 1
            else:
                acc_result["NA_fp"] += 1
        else:
            if "no answer" in ans:
                acc_result["NA_tn"] += 1
            else:
                acc_result["NA_fn"] += 1
    return acc_result

def squad_metric(predict, answers, acc_result, tokenizer):
    if acc_result is None:
        acc_result = {"train": False, "total": 0, "em_sum": 0, "f1_sum": 0., "NA_tp": 0, "NA_fp": 0, "NA_tn": 0, "NA_fn": 0}
    pred = []
    for p in predict:
        tmp = []
        for n in p:
            if n == 1:
                break
            tmp.append(int(n))
        pred.append(normalize_answer(tokenizer.decode(tmp, skip_special_tokens=True)))
    # pred = [normalize_answer([int(n) for n in p if n == 1 break], skip_special_tokens=True)) for p in predict]
    ground = [{normalize_answer(a) for a in ans} for ans in answers]

    # print(pred)
    # print(ground)
    # print("==" * 10)
    acc_result["em_sum"] += squad_em(pred, ground)
    acc_result["f1_sum"] += squad_f1(pred, ground)
    acc_result["total"] += len(pred)
    acc_result = squad_NAF1(pred, ground, acc_result)
    # print(acc_result)
    return acc_result