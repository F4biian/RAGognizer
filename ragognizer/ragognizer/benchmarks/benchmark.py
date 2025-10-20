import copy
import gc
import random
from typing import Any, Literal
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, average_precision_score, f1_score, confusion_matrix, roc_auc_score, auc

from ragognizer.detectors.detector import HallucinationDetector

class Benchmark:
    def __init__(
        self,
        name: str,
        *benchmarks,
        ignore_post_hallucination: bool=False,
        max_entries: int = None,
    ):
        self.name = name
        self.benchmarks = benchmarks
        self.ignore_post_hallucination = ignore_post_hallucination
        self.max_entries = max_entries

    def _limit(self, entries: list) -> list:
        if self.max_entries is None or self.max_entries >= len(entries):
            return entries
        return list(random.Random(432).sample(entries, self.max_entries))

    def _aggregate_prediction_to_response_level(
        self,
        pred: list[dict[str, str | int | float]] | None
    ) -> dict[str, int | float] | None:
        if not pred:
            return None
        max_pred = max(pred, key=lambda a: a["pred"] if a["pred"] is not None else np.nan)["pred"]
        max_prob = max(pred, key=lambda a: a["prob"] if a["prob"] is not None else np.nan)["prob"]
        return {
            "prob": max_prob,
            "pred": max_pred,
        }

    def _aggregate_annotations_to_response_level(
        self,
        token_level_entries: list[dict[str, list[dict[str, str]] | str | list[str] | list[str, str | int]]]
    ) -> None:
        # Override annotations list with maximum label
        for entry in token_level_entries:
            max_label = max(entry["annotations"], key=lambda a: a["label"])
            entry["annotations"] = max_label["label"]

    def _fill_non_hallu_annotations(
        self,
        response: str,
        hallucinations: list[dict[str, str | int]],
        start_key: str = "start",
        end_key: str = "end",
        text_key: str = "text",
    ) -> list[dict[str, int | str]]:
        hallucinations = [{"start": h[start_key], "end": h[end_key], "text": h[text_key]} for h in hallucinations]

        # Assert each character of `response` the value 1 (hallucination) or 0 (no hallucination)
        label_by_char = np.zeros(shape=len(response), dtype=np.int32)
        for h in hallucinations:
            h_start = h["start"]
            h_end = h["end"]
            label_by_char[h_start:h_end] = 1
            h["label"] = 1

        # Find groups with label 0 (no hallucination)
        no_hallucinations = []
        last_label = 1
        last_start = 0
        for i in range(len(label_by_char)):
            curr_label = label_by_char[i]
            # If last char was hallucinated and current char is not -> new group starts
            if curr_label != last_label and curr_label == 0:
                last_start = i
            
            is_end_of_response = (curr_label == 0 and len(label_by_char)-1 == i)
            if is_end_of_response:
                # Increase by one, so that the last char of the response is also covered
                i += 1

            # If last char was not hallucinated and current char is (or end of response) -> new group finishes
            if (curr_label != last_label and curr_label == 1) or is_end_of_response:
                no_hallucinations.append({
                    "start": last_start,
                    "end": i,
                    "text": response[last_start:i],
                    "label": 0,
                })

            last_label = curr_label

        all_annotations = hallucinations + no_hallucinations

        # Sort by start index
        all_annotations = sorted(
            all_annotations,
            key=lambda entry: entry["start"]
        )

        return all_annotations

    def _metrics_from_arrays(
        self,
        true_labels: np.array,
        pred_labels: np.array,
        pred_probs: np.array,
    ) -> dict:
        labels_mask = ~np.isnan(pred_labels.astype(np.float32))
        probs_mask  = ~np.isnan(pred_probs.astype(np.float32))

        if np.sum(labels_mask) == 0:
            precision = np.nan
            recall = np.nan
            f1 = np.nan
            conf_matrix = []
        else:
            precision = precision_score(y_true=true_labels[labels_mask], y_pred=pred_labels[labels_mask])
            recall = recall_score(y_true=true_labels[labels_mask], y_pred=pred_labels[labels_mask])
            f1 = f1_score(y_true=true_labels[labels_mask], y_pred=pred_labels[labels_mask])
            conf_matrix = confusion_matrix(true_labels[labels_mask], pred_labels[labels_mask]).tolist()

        if np.sum(probs_mask) == 0:
            area_rr_hr, area_rr_tpr, hrs, rej_rates, tprs = np.nan, np.nan, np.empty(shape=(0,)), np.empty(shape=(0,)), np.empty(shape=(0,))
            auroc = np.nan
            ap = np.nan
        else:
            area_rr_hr, area_rr_tpr, hrs, rej_rates, tprs = self._hallurate_vs_rejrate_curve(true_labels[probs_mask], pred_probs[probs_mask])
            auroc = roc_auc_score(y_true=true_labels[probs_mask], y_score=pred_probs[probs_mask])
            ap = average_precision_score(y_true=true_labels[probs_mask], y_score=pred_probs[probs_mask])

        return {
            "y_trues": true_labels.tolist(),
            "y_preds": pred_labels.tolist(),
            "y_scores": pred_probs.tolist(),

            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix,

            "label_mean": np.mean(true_labels), # baseline for auprc
            "auprc": ap,
            "auroc": auroc,

            "auc_rr_hr": float(area_rr_hr),
            "auc_rr_tpr": float(area_rr_tpr),
            "hallucination_rates": hrs.tolist(),
            "rejection_rates": rej_rates.tolist(),
            "tprs": tprs.tolist(),
        }

    def _metrics_from_subbenchmarks(
        self,
        results_by_bm: dict,
    ) -> dict:
        rl_true_labels = []
        rl_pred_labels = []
        rl_pred_probs  = []

        tl_true_labels = []
        tl_pred_labels = []
        tl_pred_probs  = []

        for key in results_by_bm:
            metrics = results_by_bm[key]["metrics"]

            rl_metrics = metrics["response_level"]
            tl_metrics = metrics["token_level"]

            if rl_metrics:
                rl_true_labels.extend(rl_metrics["y_trues"])
                rl_pred_labels.extend(rl_metrics["y_preds"])
                rl_pred_probs.extend(rl_metrics["y_scores"])
            if tl_metrics:
                tl_true_labels.extend(tl_metrics["y_trues"])
                tl_pred_labels.extend(tl_metrics["y_preds"])
                tl_pred_probs.extend(tl_metrics["y_scores"])
        
        new_metrics = {
            "response_level": self._metrics_from_arrays(
                true_labels=np.array(rl_true_labels),
                pred_labels=np.array(rl_pred_labels),
                pred_probs=np.array(rl_pred_probs),
            ) if rl_true_labels else None,
            "token_level": self._metrics_from_arrays(
                true_labels=np.array(tl_true_labels),
                pred_labels=np.array(tl_pred_labels),
                pred_probs=np.array(tl_pred_probs),
            ) if tl_true_labels else None,
        }

        return new_metrics

    def _hallurate_vs_rejrate_curve(self, labels, probs):
        labels = labels.astype(np.int64)

        thresholds = np.unique(probs)
        thresholds = np.sort(thresholds)

        rej_rates = []
        hrs       = []
        tprs      = []

        all_hallucinations = np.sum(labels)

        for t in thresholds:
            preds = (probs >= t).astype(np.int64)

            # Hallucinations still coming through after MLP as a filter
            fn = np.sum((preds == 0) & (labels == 1))

            # Everything that comes through
            coming_through_n = np.sum(preds == 0)

            if coming_through_n <= 0:
                continue

            # Hallucinations still coming through relative to all coming through (hallucination rate)
            hr = fn / coming_through_n
            tpr = 1 - fn / all_hallucinations

            rej_rate = 1 - coming_through_n / preds.shape[0]

            hrs.append(hr)
            rej_rates.append(rej_rate)
            tprs.append(tpr)

        sorted_points = sorted(zip(rej_rates, hrs, tprs))
        rej_rates, hrs, tprs = zip(*sorted_points)

        rej_rates = np.array(rej_rates)
        hrs = np.array(hrs)
        tprs = np.array(tprs)

        area_rr_hr = auc(rej_rates, hrs)
        area_rr_tpr = auc([0] + list(rej_rates) + [1], [0] + list(tprs) + [1])

        return area_rr_hr, area_rr_tpr, hrs, rej_rates, tprs

    def _calculate_metrics(
        self,
        results: list[dict[str, Any]],
        token_level: bool,
    ) -> dict[str, Any] | None:
        if not results:
            return None
        
        true_labels = []
        pred_labels = []
        pred_probs  = []

        # Tokens are determined by detector (if token level)
        for entry in results:
            pred = entry["prediction"]
            annotations = entry["annotations"]
            if token_level:
                # Assert each character of `response` the value 1 (hallucination) or 0 (no hallucination)
                label_by_char = np.zeros(shape=len(entry["response"]), dtype=np.int32)
                for span in annotations:
                    label_by_char[span["start"]:span["end"]] = span["label"]
                last_label = None
                for tok in pred:
                    if tok["end"] - tok["start"] > 0:
                        true_label = np.max(label_by_char[tok["start"]:tok["end"]])

                        if self.ignore_post_hallucination:
                            if last_label == 1 and true_label == 0:
                                break
                            last_label = true_label

                        true_labels.append(true_label)
                        pred_labels.append(tok["pred"])
                        pred_probs.append(tok["prob"])
            else:
                true_labels.append(annotations)
                pred_labels.append(pred["pred"])
                pred_probs.append(pred["prob"])

        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        pred_probs  = np.array(pred_probs)

        return self._metrics_from_arrays(true_labels, pred_labels, pred_probs)

    def run_on_benchmark(
        self,
        benchmark: "Benchmark",
        detector: HallucinationDetector,
    ) -> tuple[dict[Literal["response_level", "token_level"], list[dict]], dict[Literal["response_level", "token_level"], dict[str, Any]] | None]:
        results = {
            "response_level": [],
            "token_level": [],
        }
        token_level_entries = benchmark.get_entries(token_level=True)

        data_to_aggregate = False

        if token_level_entries:
            for entry in tqdm(token_level_entries, desc=f"{detector.name} | {benchmark.name} (Token-Level)"):
                new_entry = copy.deepcopy(entry)
                del new_entry["annotations"]
                try:
                    prediction = detector.predict(**new_entry, token_level=True)
                except Exception as ex:
                    prediction = None
                    print(f"WARNING: Failed to predict on token level due to the following error: {ex}")
                    torch.cuda.empty_cache()
                    gc.collect()

                if prediction is not None:
                    data_to_aggregate = True

                results["token_level"].append({
                    **entry,
                    "prediction": prediction
                })
                torch.cuda.empty_cache()
                gc.collect()

        response_level_entries = benchmark.get_entries(token_level=False)

        if detector.aggregate_response_level_from_token_level and data_to_aggregate:
            if response_level_entries:
                for i in tqdm(range(len(results["token_level"])), desc=f"{detector.name} | {benchmark.name} (Token-Level + Agg. Response-Level)"):
                    if results["token_level"][i]["user"] != response_level_entries[i]["user"]:
                        raise Exception("Response-level data must align with token-level data, but two entries for `user` do not match.")
                    if results["token_level"][i]["response"] != response_level_entries[i]["response"]:
                        raise Exception("Response-level data must align with token-level data, but two entries for `response` do not match.")
                    
                    pred = copy.deepcopy(results["token_level"][i]["prediction"])
                    pred = self._aggregate_prediction_to_response_level(pred)

                    results["response_level"].append({
                        **response_level_entries[i],
                        "prediction": pred
                    })
        else:
            if response_level_entries:
                for entry in tqdm(response_level_entries, desc=f"{detector.name} | {benchmark.name} (Response-Level)"):
                    new_entry = copy.deepcopy(entry)
                    del new_entry["annotations"]
                    try:
                        prediction = detector.predict(**new_entry, token_level=False)
                    except Exception as ex:
                        prediction = None
                        print(f"WARNING: Failed to predict on token level due to the following error: {ex}")
                        torch.cuda.empty_cache()
                        gc.collect()

                    results["response_level"].append({
                        **entry,
                        "prediction": prediction
                    })
                    torch.cuda.empty_cache()
                    gc.collect()

        # Remove entries that have None as prediction (level not supported by detector or error)
        if results["token_level"]:
            results["token_level"] = list(filter(lambda e: e["prediction"] is not None, results["token_level"]))
        if results["response_level"]:
            results["response_level"] = list(filter(lambda e: e["prediction"] is not None, results["response_level"]))

        metrics = {
            "token_level":    self._calculate_metrics(results=results["token_level"], token_level=True),
            "response_level": self._calculate_metrics(results=results["response_level"], token_level=False),
        }

        return results, metrics

    def run(
        self,
        detector: HallucinationDetector,
    ) -> dict[str, dict[Literal["response_level", "token_level"], list[dict]]]:
        # Run the group of benchmarks if there is a group
        if len(self.benchmarks) > 0:
            results_by_bm = {}
            for bm in self.benchmarks:
                results, metrics = self.run_on_benchmark(
                    benchmark=bm,
                    detector=detector,
                )
                results_by_bm[bm.name] = {
                    "results": results,
                    "metrics": metrics
                }
            results_by_bm["All"] = {
                "metrics": self._metrics_from_subbenchmarks(results_by_bm=results_by_bm)
            }
            return {
                self.name: results_by_bm
            }
        # Run the benchmark itself if there is no group
        else:
            results, metrics = self.run_on_benchmark(
                benchmark=self,
                detector=detector,
            )
            return {
                self.name: {
                    "results": results,
                    "metrics": metrics
                }
            } 
        
    def get_entries(
        self,
        token_level: bool
    ) -> list[dict[str, str | list[str]]] | None:
        pass