import json
import logging
import os
import re
import string
import time
from argparse import Namespace
from collections import Counter, defaultdict
from dataclasses import fields
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import disable_caching
from peft import get_peft_model
from tqdm.auto import tqdm
from transformers import (
    PreTrainedModel,
    ProgressCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import has_length

from ctx_to_lora.data.collator import eval_collator, generation_collator
from ctx_to_lora.data.definitions import (
    CLOSED_QA_DATASETS,
    CTX_AFFIXES,
    LONGBENCH_E_TASKS,
    LONGBENCH_TASKS,
    MULTI_ANSWER_DATASETS,
)
from ctx_to_lora.data.processing import (
    get_tokenized_dataset,
    load_answers,
)
from ctx_to_lora.data.self_gen_template import SELF_QA_INTX
from ctx_to_lora.metrics import (
    LENGTH_BINS,
    Evaluator,
    compute_metrics,
    compute_per_token_acc,
    compute_perplexity,
    compute_prefix_matching,
    compute_rouge,
)
from ctx_to_lora.model_loading import (
    get_lora_config,
    get_model,
    get_model_and_tokenizer,
    get_tokenizer,
)
from ctx_to_lora.modeling.context_distillation import CtxDistillModel
from ctx_to_lora.modeling.generative_adapter import GenerativeAdapter
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.modeling.relevance_gate import (
    RelevanceGatedModel,
    build_relevance_gate,
)
from ctx_to_lora.modeling.text_to_lora import TextToLoRA
from ctx_to_lora.tracker.tracker import (
    add_tracker,
    print_global_tracker_stats,
    print_tracker_stats,
    reset_trackers,
    save_tracker_stats_csv,
)
from ctx_to_lora.utils import clear_gpu, concat_list, get_run_name, setup_logging

logger = logging.getLogger()


class RankZeroProgressCallback(ProgressCallback):
    """Trainer progress bar with an explicit rank and task label."""

    def on_prediction_step(
        self, args, state, control, eval_dataloader=None, **kwargs
    ):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader),
                    desc="rank0 evaluation",
                    leave=self.training_bar is None,
                    dynamic_ncols=True,
                )
            self.prediction_bar.update(1)


REFUSAL_PATTERNS = (
    r"\bcannot answer\b",
    r"\bcan not answer\b",
    r"\bunable to answer\b",
    r"\bcannot be answered\b",
    r"\bdoes not mention\b",
    r"\bdoesn't mention\b",
    r"\bnot mentioned\b",
    r"\bnot provided\b",
    r"\bno information\b",
    r"\binsufficient information\b",
    r"\bnot relevant to the question\b",
    r"\bnot related to the question\b",
    r"\bbased on the (?:given|provided) (?:text|context|information).{0,80}"
    r"\b(?:cannot|can't|unable)\b",
)


def compute_refusal_rate(pred_texts: list[str]) -> tuple[float, list[bool]]:
    """Return the fraction of answers that refuse or flag irrelevant context."""
    refusal_flags = [
        any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in REFUSAL_PATTERNS)
        for text in pred_texts
    ]
    return float(np.mean(refusal_flags)) if refusal_flags else 0.0, refusal_flags


def save_eval_summary(metrics: dict[str, float], output_dir: str) -> None:
    """Save the requested evaluation metrics and their bar chart."""
    metric_order = (
        "rougeL.recall",
        "rougeL.precision",
        "rougeL.f1",
        "refusal_rate",
    )
    summary = {name: float(value) for name, value in metrics.items()}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "evaluation_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    summary_df = pd.DataFrame(
        [{"metric": name, "value": value} for name, value in summary.items()]
    )
    summary_df.to_csv(output_path / "evaluation_metrics.csv", index=False)
    summary_df.to_csv(output_path / "evaluation_results_generation.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "ROUGE-L Recall",
        "ROUGE-L Precision",
        "ROUGE-L F1",
        "Refusal Rate",
    ]
    values = [summary[name] for name in metric_order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        labels,
        values,
        color=["#2878B5", "#D95F02", "#2A9D8F", "#C44E52"],
    )
    ax.set_title("Evaluation Results")
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(1.0, max(values) * 1.2))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_path / "evaluation_metrics.png", dpi=200)
    plt.close(fig)


def get_llmlingua_model_class():
    try:
        from ctx_to_lora.modeling.llm_lingua import LLMLinguaModel
    except ModuleNotFoundError as exc:
        if exc.name == "llmlingua":
            raise ModuleNotFoundError(
                "LLMLingua is optional. Install `llmlingua>=0.2.2` only when "
                "running with --use_llmlingua."
            ) from exc
        raise
    return LLMLinguaModel


# from https://gist.github.com/cloneofsimo/8abd0284d4738f28f04200628f9a83f5
# https://github.com/Nordth/humanize-ai-lib/blob/main/src/humanize-string.ts

_HIDDEN_CHARS = re.compile(
    r"[\u00AD\u180E\u200B-\u200F\u202A-\u202E\u2060\u2066-\u2069\uFEFF]"
)
_TRAILING_WS = re.compile(r"[ \t\x0B\f]+$", re.MULTILINE)
_NBSP = re.compile(r"\u00A0")
_DASHES = re.compile(r"[—–]+")  # em- & en-dashes → ASCII hyphen
_DQUOTES = re.compile(r"[“”«»„]")  # curly / guillemets → "
_SQUOTES = re.compile(r"[‘’ʼ]")  # curly apostrophes → '
_ELLIPSIS = re.compile(r"…")  # single‐char ellipsis → "..."
_ENDASH = re.compile(r"\u2013")
_EMDASH = re.compile(r"\u2014")


def humanize_str(text: str) -> str:
    text = _HIDDEN_CHARS.sub("", text)
    text = _TRAILING_WS.sub("", text)
    text = _NBSP.sub(" ", text)
    text = _DASHES.sub("-", text)
    text = _ENDASH.sub("-", text)
    text = _EMDASH.sub("-", text)
    text = _DQUOTES.sub('"', text)
    text = _SQUOTES.sub("'", text)
    text = _ELLIPSIS.sub("...", text)
    return text


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        # return "".join(ch for ch in text if ch not in exclude)
        return " ".join(re.split(f"[{string.punctuation}]", text))

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(humanize_str(s)))))


def split_string(s: str) -> list[str]:
    out = re.split(r"[- \s]", s)  # split by hyphen, space, or whitespace
    return [x for x in out if x]  # remove empty spaces


def f1_score(
    prediction: list[str], ground_truth: list[str]
) -> tuple[float, float, float]:
    """Compute F1 score, precision, and recall between prediction and ground truth strings."""
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction) if len(prediction) > 0 else 0
    recall = 1.0 * num_same / len(ground_truth) if len(ground_truth) > 0 else 0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return f1, precision, recall


def compute_qa_f1_score(
    pred_texts: list[str], answers_list: list[list[str]]
) -> dict[str, float]:
    """
    Word-level F1 score for evaluating question answering systems.
    Order of the words does not matter.
    """
    f1_scores = []
    precisions = []
    recalls = []

    for prediction, answers in zip(pred_texts, answers_list):
        normalized_prediction = normalize_answer(prediction)
        prediction_words = split_string(normalized_prediction)
        best_f1 = 0
        best_precision = 0
        best_recall = 0

        for answer in answers:
            normalized_label = normalize_answer(answer)
            label_words = split_string(normalized_label)
            f1, precision, recall = f1_score(prediction_words, label_words)
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall

        f1_scores.append(best_f1)
        precisions.append(best_precision)
        recalls.append(best_recall)

    return dict(
        qa_f1_score=np.mean(f1_scores),
        qa_precision=np.mean(precisions),
        qa_recall=np.mean(recalls),
    ), dict(qa_f1_score=f1_scores, qa_precision=precisions, qa_recall=recalls)


def add_longbench_tasks(ds_names: list[str]) -> None:
    """Add longbench tasks to dataset names list."""
    if "longbench" in ds_names:
        ds_names.remove("longbench")
        ds_names += LONGBENCH_TASKS
    if "longbench_e" in ds_names:
        ds_names.remove("longbench_e")
        ds_names += LONGBENCH_E_TASKS


def save_generated_text(
    samples: list[dict],
    per_sample_metric: dict[str, list[float]],
    output_dir: str,
    split: str,
) -> None:
    """Save generated text samples to JSONL file."""
    os.makedirs(output_dir, exist_ok=True)
    # Create any necessary subdirectories if split contains path separators
    if "/" in split:
        split_dir = os.path.join(output_dir, os.path.dirname(split))
        os.makedirs(split_dir, exist_ok=True)

    metric_keys = list(per_sample_metric.keys())
    # assert len(metric_keys) == 1

    with open(f"{output_dir}/{split}_generated_text.jsonl", "w") as f:
        for i, sample in enumerate(samples):
            for metric_name in metric_keys:
                sample[f"{metric_name}"] = per_sample_metric[metric_name][i]
            f.write(json.dumps(sample) + "\n")


# ============================================================================
# CSV Export Utilities
# ============================================================================


def _extract_model_info(eval_trainer) -> tuple[str, bool]:
    """Extract model name and type information from the trainer."""
    model_name = "unknown_model"
    is_hypernet = False

    if hasattr(eval_trainer.model, "base_model"):
        if hasattr(eval_trainer.model.base_model, "config"):
            model_name = getattr(
                eval_trainer.model.base_model.config,
                "name_or_path",
                getattr(
                    eval_trainer.model.base_model.config, "_name_or_path", "unknown"
                ),
            )
        is_hypernet = hasattr(eval_trainer.model, "ctx_encoder")
    elif hasattr(eval_trainer.model, "config"):
        model_name = getattr(
            eval_trainer.model.config,
            "name_or_path",
            getattr(eval_trainer.model.config, "_name_or_path", "unknown"),
        )

    # Clean up model name for display
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    if getattr(eval_trainer.args, "run_name", None):
        model_name += f"_{eval_trainer.args.run_name}"

    return model_name, is_hypernet


def _parse_metrics_for_csv(
    metrics_dict: dict[str, dict[str, any]],
) -> tuple[set, set, set]:
    """Parse metrics dictionary to extract unique metrics, length groups, and splits."""
    all_metrics = set()
    all_length_groups = set()
    all_splits = set()

    for split_name, metrics in metrics_dict.items():
        all_splits.add(split_name)

        for metric_key in metrics.keys():
            if not metric_key.startswith(split_name):
                continue

            # Skip timing and performance metrics that aren't evaluation results
            if any(
                skip_term in metric_key
                for skip_term in [
                    "model_preparation_time",
                    "steps_per_second",
                    "samples_per_second",
                    "runtime",
                ]
            ):
                continue

            # Remove the split prefix to get the actual metric name
            metric_name = metric_key[len(split_name) + 1 :]

            # Check if this is a length-specific metric
            if "_len_" in metric_name:
                base_metric, length_part = metric_name.split("_len_", 1)
                all_metrics.add(base_metric)
                all_length_groups.add(length_part)
            else:
                all_metrics.add(metric_name)

    # Add overall metric (no length grouping)
    all_length_groups.add("overall")
    return all_metrics, all_length_groups, all_splits


def _sort_length_groups(length_groups: set[str]) -> list[str]:
    """Sort length groups with custom ordering for proper numerical ranges."""

    def sort_key(length_group: str) -> tuple:
        if length_group == "overall":
            return (1, 0, 0)  # Put "overall" after numerical ranges
        try:
            # Parse "low-high" format
            low, high = map(float, length_group.split("-"))
            return (0, low, high)
        except (ValueError, IndexError):
            return (2, 0, 0)  # Put any malformed strings last

    return sorted(list(length_groups), key=sort_key)


def create_metrics_csv(
    metrics_dict: dict[str, dict[str, any]],
    output_dir: str,
    model_name: str,
    is_hypernet_model: bool = False,
    remove_context: bool = False,
    csv_suffix: str = "",
) -> None:
    """
    Create a CSV file with columns: model_name, group_len, tasks, num_samples, and all available metrics.
    One row per model-length-task combination.

    Args:
        metrics_dict: Dictionary containing evaluation metrics for each dataset/split
        output_dir: Directory to save the CSV file
        model_name: Name of the model being evaluated
        is_hypernet_model: Whether this is a hypernet/modulated model or base model
        remove_context: Whether context was removed during evaluation
        csv_suffix: Additional suffix for the CSV filename
    """
    os.makedirs(output_dir, exist_ok=True)

    # Parse metrics to extract components
    all_metrics, all_length_groups, all_splits = _parse_metrics_for_csv(metrics_dict)

    # Sort for consistent ordering
    all_length_groups = _sort_length_groups(all_length_groups)
    all_splits = sorted(all_splits)

    # Create rows for each model-length-task combination
    rows = []

    for task in all_splits:
        metrics = metrics_dict[task]

        for length_group in all_length_groups:
            # Initialize row with basic info
            row = {
                "model_name": model_name,
                "group_len": length_group,
                "tasks": task,
                "num_samples": 0,
            }

            # Look for all metrics for this task and length group
            if length_group == "overall":
                # Look for overall metrics (no length suffix)
                for metric_key in metrics:
                    if not metric_key.startswith(f"{task}_"):
                        continue

                    # Skip length-specific metrics
                    if "_len_" in metric_key:
                        continue

                    # Extract metric name after task prefix
                    metric_name = metric_key[len(task) + 1 :]
                    if metric_name in [
                        "samples_per_second",
                        "steps_per_second",
                        "model_preparation_time",
                        "runtime",
                    ]:
                        # Skip timing and performance metrics
                        continue

                    # Handle num_samples specially
                    if metric_name.startswith("num_samples_"):
                        row["num_samples"] = metrics[metric_key]
                    else:
                        # Add all other metrics as columns
                        row[metric_name] = metrics[metric_key]
            else:
                # Look for length-specific metrics
                for metric_key in metrics:
                    if not metric_key.startswith(f"{task}_"):
                        continue

                    # Only process metrics for this specific length group
                    if f"_len_{length_group}" not in metric_key:
                        continue

                    # Extract metric name (remove task prefix and length suffix)
                    metric_part = metric_key[len(task) + 1 :]
                    metric_name = metric_part.replace(f"_len_{length_group}", "")
                    if metric_name in [
                        "samples_per_second",
                        "steps_per_second",
                        "model_preparation_time",
                        "runtime",
                    ]:
                        # Skip timing and performance metrics
                        continue

                    # Handle num_samples specially
                    if metric_name.startswith("num_samples_"):
                        row["num_samples"] = metrics[metric_key]
                    else:
                        # Add all other metrics as columns
                        row[metric_name] = metrics[metric_key]

            # Fill missing metrics with N/A for consistent columns
            for metric in all_metrics:
                if metric not in row and not metric.startswith("num_samples"):
                    row[metric] = "N/A"

            rows.append(row)

    # Create DataFrame
    if rows:
        new_df = pd.DataFrame(rows)
        # define categories so that they're sorted properly
        new_df["group_len"] = pd.Categorical(
            new_df["group_len"],
            categories=[f"{i}-{j}" for (i, j) in LENGTH_BINS] + ["overall"],
        )

        # Sort by tasks, then by length group
        new_df = new_df.sort_values(["tasks", "group_len"]).reset_index(drop=True)

        # Construct filename
        csv_filename = "evaluation_results"
        if csv_suffix:
            csv_filename += f"_{csv_suffix}"
        if remove_context:
            csv_filename += "_no_context"
        csv_filename += ".csv"

        csv_path = os.path.join(output_dir, csv_filename)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Check if CSV already exists and merge if it does
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)

                # Remove existing rows with the same model_name, group_len, and tasks
                # to avoid duplicates when updating
                mask = ~existing_df["tasks"].isin(new_df["tasks"])
                existing_df = existing_df[mask]

                # Concatenate existing and new data
                df = pd.concat([existing_df, new_df], ignore_index=True)

                # Sort by tasks, then by length group
                df = df.sort_values(["tasks"]).reset_index(drop=True)

                print(f"Updated existing CSV with {len(new_df)} new rows")
            except Exception as e:
                print(f"Warning: Could not read existing CSV ({e}), creating new file")
                df = new_df
        else:
            df = new_df
            print(f"Created new CSV with {len(new_df)} rows")

        df.to_csv(csv_path, index=False)

        print(f"Evaluation results saved to: {csv_path}")
    else:
        print("No evaluation data found to save to CSV")


# ============================================================================
# Evaluation Functions
# ============================================================================


def decode_test_result(
    test_dataset, test_result, tokenizer, ctx_tokenizer
) -> list[dict]:
    """Decode test results into human-readable format."""
    out = []
    for sample, pred_toks in zip(test_dataset, test_result.predictions):
        d = dict()
        if "labels" in sample:
            start_idx = np.argmax(sample["labels"] != -100)
            label_toks = sample["labels"][start_idx:]
            # labels are padded with -100, so we need to
            # replace them with the pad token id
            label_toks = np.where(
                label_toks == -100, tokenizer.pad_token_id, label_toks
            )
            label_text = tokenizer.decode(label_toks, skip_special_tokens=True)
            d["label"] = label_text.strip()

        # remove the label part
        input_toks = sample["input_ids"][:start_idx]

        gen_toks = pred_toks[np.argmax(pred_toks != tokenizer.pad_token_id) :]
        # gen_toks = gen_toks[start_idx:]
        suffix = np.array(CTX_AFFIXES[tokenizer.name_or_path]["suffix"])
        # iterate over gen_toks and take the answer after the suffix
        for i in range(len(gen_toks) - len(suffix), -1, -1):
            if all(gen_toks[i : i + len(suffix)] == suffix):
                gen_toks = gen_toks[i + len(suffix) :]
                break
        gen_toks = np.where(gen_toks == -100, tokenizer.pad_token_id, gen_toks)

        d["input"] = tokenizer.decode(input_toks, skip_special_tokens=False)
        d["generated"] = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
        if "ctx_ids" in sample:
            d["context"] = ctx_tokenizer.decode(
                concat_list(sample["ctx_ids"]), skip_special_tokens=False
            )
        for k in sample:
            if k.endswith("_len"):
                d[k] = sample[k].item()
        out.append(d)

    # sort samples by length if possible
    len_key = "ctx_ids_len" if "ctx_ids_len" in sample else "input_ids_len"
    sorted(out, key=lambda x: x[len_key])

    return out


def eval_generation(
    eval_trainer,
    tokenizer,
    ctx_tokenizer,
    datasets,
    answers,
    split,
    remove_context,
    gen_kwargs,
) -> dict[str, dict]:
    """Evaluate model using generation and save metrics to CSV."""
    if not isinstance(datasets, dict):
        datasets = {"": datasets}

    out = {}
    for ds_name, ds in datasets.items():
        print(f"Evaluating: {ds_name}")
        split_name = f"{split}_{ds_name}" if ds_name else split
        if remove_context:
            split_name += "_no_context"

        clear_gpu()
        eval_result = eval_trainer.predict(
            ds,
            metric_key_prefix=split_name,
            **gen_kwargs,
        )
        decoded_txts = decode_test_result(ds, eval_result, tokenizer, ctx_tokenizer)

        pred_texts = [txt["generated"] for txt in decoded_txts]
        label_texts = [txt["label"] for txt in decoded_txts]
        if ds_name in answers:
            answers_list = answers[ds_name]["answers"]
        else:
            answers_list = [[txt] for txt in label_texts]
        n = len(pred_texts)

        print("Computing ROUGE-L Recall, Precision, and F1")
        rouge_metrics, per_sample_metric = compute_rouge(pred_texts, answers_list)
        refusal_rate, refusal_flags = compute_refusal_rate(pred_texts)
        per_sample_metric["refusal"] = refusal_flags
        summary_metrics = {
            **rouge_metrics,
            "refusal_rate": refusal_rate,
            "num_examples": n,
        }

        concise_metrics = {
            f"{split_name}_{name}": value for name, value in summary_metrics.items()
        }
        eval_result.metrics.clear()
        eval_result.metrics.update(concise_metrics)

        out[split_name] = eval_result.metrics
        if eval_trainer.is_world_process_zero():
            save_generated_text(
                decoded_txts,
                per_sample_metric,
                split=split_name,
                output_dir=eval_trainer.args.output_dir,
            )
            eval_trainer.log_metrics(split_name, eval_result.metrics)
            eval_trainer.save_metrics(split_name, eval_result.metrics)
            save_eval_summary(
                summary_metrics,
                output_dir=eval_trainer.args.output_dir,
            )
        clear_gpu()

    return out


def eval_teacher_forcing(
    eval_trainer, datasets, split, remove_context
) -> dict[str, dict]:
    """Evaluate using teacher forcing and save metrics to CSV."""
    if not isinstance(datasets, dict):
        datasets = {"": datasets}

    out = {}
    for ds_name, ds in datasets.items():
        split_name = f"{split}_{ds_name}" if ds_name else split
        if "/" in split_name:
            split_dir = os.path.join(
                eval_trainer.args.output_dir, os.path.dirname(split_name)
            )
            os.makedirs(split_dir, exist_ok=True)
        if remove_context:
            split_name += "_no_context"

        metrics = eval_trainer.evaluate(ds, metric_key_prefix=split_name)
        out[split_name] = metrics
        eval_trainer.log_metrics(split_name, metrics)
        eval_trainer.save_metrics(split_name, metrics)
        clear_gpu()

    # Create CSV for teacher forcing metrics
    if out:
        model_name, is_hypernet = _extract_model_info(eval_trainer)
        create_metrics_csv(
            out,
            output_dir=eval_trainer.args.output_dir,
            model_name=model_name,
            is_hypernet_model=is_hypernet,
            remove_context=remove_context,
            csv_suffix="teacher_forcing",
        )

    return out


def evaluate(
    checkpoint_path: str,
    model_name_or_path: str,
    eval_batch_size: int,
    args: Namespace,
    split: str,
    max_ctx_chunk_len: int,
    max_new_tokens: int,
    generative: bool,
) -> dict[str, dict]:
    """Main evaluation function."""
    assert split in ["validation", "test"]
    ctx_name = None
    model_kwargs = dict(attn_implementation="flash_attention_2")

    tokenizer = get_tokenizer(args.model_name_or_path, train=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    canonical_model_name = getattr(args, "canonical_model_name_or_path", None)
    if canonical_model_name in CTX_AFFIXES:
        CTX_AFFIXES[tokenizer.name_or_path] = CTX_AFFIXES[canonical_model_name]
        template_path = (
            Path(__file__).resolve().parents[2]
            / "chat_templates"
            / f"{canonical_model_name}.jinja"
        )
        if template_path.exists():
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")

    use_cd = False
    ctx_model_max_len = None
    base_model = None

    if model_name_or_path is None:
        try:
            # Checkpoints may have been saved from cuda:0. Loading on CPU avoids
            # rank 1 creating an unnecessary CUDA context on rank 0's GPU.
            state_dict = torch.load(
                checkpoint_path,
                weights_only=False,
                map_location="cpu",
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found. ")
        if base_model_path := getattr(args, "base_model_path", None):
            state_dict["base_model_name_or_path"] = base_model_path
            state_dict["hypernet_config"].lora_config.base_model_name_or_path = (
                base_model_path
            )
            ctx_encoder_args = state_dict["ctx_encoder_args"]
            if ctx_encoder_args.ctx_encoder_model_name_or_path:
                ctx_encoder_args.ctx_encoder_model_name_or_path = base_model_path
        ctx_name = state_dict["ctx_encoder_args"].ctx_encoder_model_name_or_path

        model = ModulatedPretrainedModel.from_state_dict(
            state_dict,
            train=False,
            base_model_kwargs=model_kwargs,
            use_flash_attn=True,
            use_sequence_packing=False,  # for generation
            user_defined_scaling=args.gen_lora_scaling,
        )
        if getattr(args, "use_llmlingua", False):
            print("Using LLMLingua-2 for compressing inp")
            LLMLinguaModel = get_llmlingua_model_class()
            inp_compressor = LLMLinguaModel(
                model.base_model, tokenizer, args.llmlingua_compression_rate
            )
            model.inp_compressor = inp_compressor
        ctx_model_max_len = model.ctx_encoder.config.max_position_embeddings
        model.enable_iterative_mode(args.use_iterative_mode)
        add_tracker(model.base_model.generate, "generate")
        add_tracker(model.generate_weights, "generate_weights")
        add_tracker(model.combine_lora, "combine_lora")
        add_tracker(model.apply_lora_to_layers, "apply_lora_to_layers")
    else:
        model = base_model = get_model(
            model_name_or_path,
            train=False,
            requires_grad=False,
            model_kwargs=model_kwargs,
            use_flash_attn=True,
        )
        add_tracker(base_model.generate, "generate")
        if use_cd := getattr(args, "use_cd", False):
            peft_config = get_lora_config(
                model_name_or_path,
                lora_r=8,
                lora_dropout=0,
                target_modules=["down_proj"],
            )
            peft_config.lora_alpha = 16
            peft_model = get_peft_model(base_model, peft_config)
            sep_seq = (
                tokenizer(
                    SELF_QA_INTX.strip("\n"),
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                .input_ids[0][1:]
                .to(base_model.device)
            )
            ctx_distill_kwargs = dict(
                prefix_tokens=torch.tensor(
                    CTX_AFFIXES[model_name_or_path]["prefix"], device=base_model.device
                ),
                ctx_inp_sep_seq=sep_seq,
                pad_token_id=tokenizer.pad_token_id,
                update_iterations=args.cd_update_iterations,
                tokenizer=tokenizer,
                reprompt_ctx=args.add_ctx_to_input,
            )
            if args.cd_use_gen_q:
                q_model, q_tokenizer = get_model_and_tokenizer(
                    "google/gemma-3-4b-it",
                    train=False,
                    requires_grad=False,
                )
                ctx_distill_kwargs["q_model"] = q_model
                ctx_distill_kwargs["q_tokenizer"] = q_tokenizer
                ctx_distill_kwargs["q_gen_rounds"] = args.q_gen_rounds
                ctx_distill_kwargs["batch_size"] = args.cd_batch_size
            model = CtxDistillModel(peft_model, **ctx_distill_kwargs)

            add_tracker(model._distill_context, "distill_context")
            add_tracker(model.generate_questions, "generate_questions")
            add_tracker(model.teacher_generate, "teacher_generate")
            add_tracker(model.student_generate, "student_generate")
        elif use_llmlingua := getattr(args, "use_llmlingua", False):
            LLMLinguaModel = get_llmlingua_model_class()
            model = LLMLinguaModel(
                base_model, tokenizer, args.llmlingua_compression_rate
            )
            ctx_model_max_len = model.base_model.config.max_position_embeddings
            add_tracker(model.base_model.generate, "generate")
            add_tracker(model.compress, "prompt_compress")

        elif use_t2l := getattr(args, "use_t2l", False):
            model = TextToLoRA(
                base_model.name_or_path,
                prefix_tokens=torch.tensor(
                    CTX_AFFIXES[model_name_or_path]["prefix"], device=base_model.device
                ),
                device=base_model.device,
            )
            ctx_model_max_len = model.base_model.config.max_position_embeddings
            add_tracker(model.base_model.generate, "base_model.generate")
            add_tracker(model.generate_weights, "generate_weights")

        elif args.use_generative_adapter:
            model = GenerativeAdapter(model=base_model, tokenizer=tokenizer)
            ctx_model_max_len = base_model.config.max_position_embeddings

    if base_model is None:
        base_model = model.base_model
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    ctx_tokenizer = tokenizer
    if ctx_name:
        ctx_tokenizer = get_tokenizer(ctx_name, train=False)
        if ctx_tokenizer.pad_token_id is None:
            ctx_tokenizer.pad_token_id = ctx_tokenizer.eos_token_id

    if getattr(args, "use_relevance_gate", False):
        if not isinstance(model, ModulatedPretrainedModel):
            raise ValueError("Relevance gate is only supported for D2L checkpoints.")
        gate = build_relevance_gate(
            args.gate_backend,
            gate_model_path=getattr(args, "gate_model_path", None),
            device=model.device,
            fallback_encoder=getattr(model, "ctx_encoder", None),
            fallback_tokenizer=ctx_tokenizer,
        )
        model = RelevanceGatedModel(
            model=model,
            gate=gate,
            threshold=args.gate_threshold,
            tokenizer=tokenizer,
            ctx_tokenizer=ctx_tokenizer,
        )
        print(
            "Using relevance gate: "
            f"backend={args.gate_backend}, threshold={args.gate_threshold}, "
            f"model_path={getattr(args, 'gate_model_path', None)}"
        )

    add_ctx_to_chat = (
        (isinstance(model, PreTrainedModel) and not args.remove_context)
        or isinstance(model, CtxDistillModel)
        or args.add_ctx_to_input
    )

    _get_tokenized_dataset = partial(
        get_tokenized_dataset,
        max_qas_len=-1,
        max_qas_per_sample=1,
        max_ctx_chunk_len=max_ctx_chunk_len,
        min_ctx_chunk_len=-1,
        num_chunk_probs=None,
        max_ctx_chunk_num=None,
        base_model_max_len=model.base_model.config.max_position_embeddings,
        tokenizer=tokenizer,
        ctx_model_max_len=ctx_model_max_len,
        ctx_tokenizer=ctx_tokenizer,
        add_ctx_to_chat=add_ctx_to_chat,
        add_negative_prompt=False,
        use_kl_loss=False,
        max_new_tokens=max_new_tokens,
        set_format="pt",
        add_self_distill_template=use_cd,  # only for eval
        truncate_if_too_long_inp=args.truncate_if_too_long_inp,  # only for eval
        truncate_if_too_long_ctx=args.truncate_if_too_long_ctx,  # only for eval
        flip_ctx_inp=args.flip_ctx_inp,  # only for eval
    )

    datasets = dict()
    answers = dict()
    ds_names = args.val_ds_names if split == "validation" else args.test_ds_names
    add_longbench_tasks(ds_names)
    for ds_name in ds_names:
        datasets[ds_name] = _get_tokenized_dataset(ds_name, split)
        # handling cases where there are multiple answers
        if ds_name in MULTI_ANSWER_DATASETS:
            answers[ds_name] = load_answers(ds_name, split)

    print(f"Datasets: {datasets}")
    print(f"Answers: {answers}")

    # truncating num val samples
    max_eval_samples_per_ds = getattr(args, "max_val_samples_per_ds", 0)
    if split == "validation" and max_eval_samples_per_ds > 0:
        for ds_name, ds in datasets.items():
            if max_eval_samples_per_ds >= len(ds):
                continue
            print(
                f"Truncating validation ds {ds_name} to "
                f"{max_eval_samples_per_ds} samples"
            )
            val_indices = np.random.permutation(len(ds))[:max_eval_samples_per_ds]
            datasets[ds_name] = ds.select(val_indices)
            if ds_name in answers:
                answers[ds_name] = answers[ds_name].select(val_indices)

    max_test_samples_per_ds = getattr(args, "max_test_samples_per_ds", 0)
    if split == "test" and max_test_samples_per_ds > 0:
        for ds_name, ds in datasets.items():
            if max_test_samples_per_ds >= len(ds):
                continue
            print(
                f"Truncating test ds {ds_name} to "
                f"{max_test_samples_per_ds} samples"
            )
            test_indices = np.random.permutation(len(ds))[:max_test_samples_per_ds]
            datasets[ds_name] = ds.select(test_indices)
            if ds_name in answers:
                answers[ds_name] = answers[ds_name].select(test_indices)

    print(f"Datasets: {datasets}")
    print(f"Answers: {answers}")

    gen_kwargs = dict(
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )

    eval_trainer_args = {}

    # Copy only necessary attributes from training_args to eval_trainer_args
    seq2seq_training_args_fields = {f.name for f in fields(Seq2SeqTrainingArguments)}
    for attr, value in dict(**vars(args)).items():
        if attr in seq2seq_training_args_fields and not attr.startswith("_"):
            eval_trainer_args[attr] = value

    eval_trainer_args["eval_strategy"] = "no"
    eval_trainer_args["save_strategy"] = "no"
    eval_trainer_args["overwrite_output_dir"] = True
    eval_trainer_args["batch_eval_metrics"] = True
    eval_trainer_args["per_device_eval_batch_size"] = eval_batch_size
    eval_trainer_args["include_for_metrics"] = ["inputs"]
    eval_trainer_args["batch_eval_metrics"] = True
    eval_trainer_args["remove_unused_columns"] = False
    eval_trainer_args["bf16"] = False
    eval_trainer_args["tf32"] = False
    eval_trainer_args["use_liger_kernel"] = False
    eval_trainer_args["dataloader_num_workers"] = 0
    eval_trainer_args["dataloader_prefetch_factor"] = None
    eval_trainer_args["disable_tqdm"] = False
    eval_trainer_args["report_to"] = []

    eval_trainer_args = Seq2SeqTrainingArguments(
        **eval_trainer_args,
        predict_with_generate=generative,
        # generation_config=GenerationConfig(**gen_kwargs),
    )

    print("=" * 80 + "\n" + "Evaluating model..." + "\n" + "=" * 80)
    print(f"checkpoint_path: {checkpoint_path}")

    model.eval()

    collator = generation_collator if generative else eval_collator

    # if isinstance(model, CtxDistillModel):

    #     model.generate = partial(
    #         model.generate,
    #         ctx_inp_sep_seq=sep_seq,
    #         reset=True,
    #     )

    trainer_kwargs = {
        "model": model,
        "args": eval_trainer_args,
        "data_collator": partial(collator, tokenizer=tokenizer),
    }

    out = {}
    if not generative:
        trainer_kwargs["compute_metrics"] = partial(
            compute_metrics,
            evaluator=Evaluator(
                [compute_per_token_acc, compute_prefix_matching, compute_perplexity]
            ),
        )
        # this is insane
        # but i don't know why calling trainer.evaluate() on different datasets
        # always gives the same numbers across datasets...
        # spents a few hours on this but couldn't find the reason
        for ds_name, ds in datasets.items():
            eval_trainer = Trainer(**trainer_kwargs)
            clear_gpu()
            metrics = eval_teacher_forcing(
                eval_trainer, {ds_name: ds}, split, args.remove_context
            )
            out.update(metrics)
    else:
        eval_trainer = Seq2SeqTrainer(**trainer_kwargs)
        eval_trainer.remove_callback(ProgressCallback)
        eval_trainer.add_callback(RankZeroProgressCallback())
        for ds_name, ds in datasets.items():
            metrics = eval_generation(
                eval_trainer,
                tokenizer,
                ctx_tokenizer,
                {ds_name: ds},
                answers,
                split,
                args.remove_context,
                gen_kwargs,
            )
            out.update(metrics)

            print_tracker_stats()
            print_global_tracker_stats()
            if hasattr(eval_trainer.model, "print_gate_stats"):
                eval_trainer.model.print_gate_stats()
                eval_trainer.model.reset_gate_stats()
            ds_suffix = "_no_context" if args.remove_context else ""
            save_tracker_stats_csv(
                f"{args.logging_dir}/{split}_{ds_name}{ds_suffix}_tracked_stats.csv"
            )
            reset_trackers()

    clear_gpu()
    return out


def run_eval(
    checkpoint_path: str = None,
    model_name_or_path: str = None,
    datasets: list[str] = None,
    dataset_path: str | None = None,
    split: str = "validation",
    eval_batch_size: int = 8,
    max_val_samples_per_ds: int = -1,
    max_test_samples_per_ds: int = -1,
    max_ctx_chunk_len: int = -1,
    remove_context: bool = False,
    max_new_tokens: int = 256,
    generative: bool = False,
    use_cd: bool = False,
    cd_update_iterations: int = 10,
    cd_use_gen_q: bool = False,
    q_gen_rounds: int = 4,
    cd_batch_size: int = 16,
    use_iterative_mode: bool = False,
    use_llmlingua: bool = False,
    llmlingua_compression_rate: float = 0.9,
    use_t2l: bool = False,
    add_ctx_to_input: bool = False,
    truncate_if_too_long_inp: bool = False,
    truncate_if_too_long_ctx: bool = False,
    flip_ctx_inp: bool = False,
    gen_lora_scaling: float = 1,
    use_generative_adapter: bool = False,
    use_relevance_gate: bool = False,
    gate_backend: str = "lexical",
    gate_threshold: float = 0.35,
    gate_model_path: str | None = None,
    base_model_path: str | None = None,
    output_dir: str | None = None,
) -> None:
    """Run evaluation with the specified parameters."""
    assert bool(model_name_or_path) ^ bool(checkpoint_path), (
        "Either --model_name_or_path or --checkpoint_path must be provided"
    )
    if (use_cd or use_llmlingua or use_t2l) and eval_batch_size != 1:
        raise ValueError("When using a baseline method, eval_batch_size must be 1.")

    if use_llmlingua and add_ctx_to_input:
        raise ValueError(
            "LLMLingua always adds compressed context to input by default."
        )
    if use_generative_adapter and (
        model_name_or_path != "mistralai/Mistral-7B-Instruct-v0.2"
    ):
        raise ValueError(
            "Generative adapter is only available for Mistral-7B-Instruct-v0.2."
        )

    disable_caching()
    set_seed(42)
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        print(
            "Worker mapping: "
            f"rank={os.environ.get('RANK', '0')}, "
            f"local_rank={os.environ['LOCAL_RANK']}, "
            f"pid={os.getpid()}, "
            f"cuda_device={torch.cuda.current_device()}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
            flush=True,
        )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    os.environ["OMP_NUM_THREADS"] = "23"

    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    slurm_job_id = f"_{os.getenv('SLURM_JOB_ID')}" if os.getenv("SLURM_JOB_ID") else ""
    run_name = get_run_name(seed_str=time.strftime("%Y%m%d-%H%M%S") + slurm_job_id)

    if checkpoint_path:
        checkpoint_dir = "/".join(checkpoint_path.split("/")[:-1])
        run_dir = "/".join(checkpoint_path.split("/")[:-2])
        cur_it = int(checkpoint_path.split("checkpoint-")[1].split("/")[0])
        try:
            args = Namespace(**yaml.unsafe_load(open(f"{run_dir}/args.yaml")))
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find args.yaml in {run_dir}. ")
        print(f"checkpoint_path: {checkpoint_path}")
        print(f"run_dir: {run_dir}")

        args.output_dir = f"{run_dir}/eval-results-{cur_it}/{run_name}"
        args.logging_dir = f"{run_dir}/eval-results-{cur_it}/{run_name}"
        args.run_name = run_dir.split("/")[-1]
        args.canonical_model_name_or_path = args.model_name_or_path
        if base_model_path:
            args.base_model_path = base_model_path
            args.model_name_or_path = base_model_path
        # modulated model doesn't see ctx by default
        # but remove_context has to be false for correct file naming
        args.remove_context = False
        args.use_iterative_mode = use_iterative_mode
        if use_llmlingua:
            args.use_llmlingua = use_llmlingua
            args.llmlingua_compression_rate = llmlingua_compression_rate
        args.use_relevance_gate = use_relevance_gate
        args.gate_backend = gate_backend
        args.gate_threshold = gate_threshold
        args.gate_model_path = gate_model_path
    else:
        args = Namespace(
            model_name_or_path=model_name_or_path,
            output_dir=f"eval_results/{model_name_or_path}/{run_name}",
            logging_dir=f"eval_results/{model_name_or_path}/{run_name}",
            run_name=f"eval_results/{model_name_or_path}/{run_name}",
            val_ds_names=[],
            test_ds_names=[],
            remove_context=remove_context,
            use_relevance_gate=use_relevance_gate,
            gate_backend=gate_backend,
            gate_threshold=gate_threshold,
            gate_model_path=gate_model_path,
            canonical_model_name_or_path="Qwen/Qwen3-4B-Instruct-2507"
            if "Qwen3-4B-Instruct-2507" in model_name_or_path
            else model_name_or_path,
        )
        if use_cd:
            args.use_cd = use_cd
            args.cd_update_iterations = cd_update_iterations
            args.cd_use_gen_q = cd_use_gen_q
            args.q_gen_rounds = q_gen_rounds
            args.cd_batch_size = cd_batch_size
        if use_llmlingua:
            args.use_llmlingua = use_llmlingua
            args.llmlingua_compression_rate = llmlingua_compression_rate
        if use_t2l:
            args.use_t2l = use_t2l
        args.use_generative_adapter = use_generative_adapter
    if max_val_samples_per_ds > 0:
        args.max_val_samples_per_ds = max_val_samples_per_ds
    if max_test_samples_per_ds > 0:
        args.max_test_samples_per_ds = max_test_samples_per_ds
    args.add_ctx_to_input = add_ctx_to_input
    args.dataset_path = dataset_path
    args.gen_lora_scaling = gen_lora_scaling
    args.truncate_if_too_long_inp = truncate_if_too_long_inp
    args.truncate_if_too_long_ctx = truncate_if_too_long_ctx
    args.flip_ctx_inp = flip_ctx_inp
    if output_dir:
        args.output_dir = output_dir
        args.logging_dir = output_dir
    if int(os.environ.get("RANK", "0")) == 0:
        setup_logging(args.logging_dir)
    logger.debug(f"CMD: {' '.join(os.sys.argv)}")

    # Override dataset names if provided via CLI
    if datasets:
        if split == "validation":
            args.val_ds_names = datasets
        else:
            args.test_ds_names = datasets

    return evaluate(
        checkpoint_path,
        model_name_or_path,
        eval_batch_size,
        args,
        split,
        max_ctx_chunk_len,
        max_new_tokens,
        generative=generative,
    )
