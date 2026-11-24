"""Fixed embeddings, correctness classifiers, validation selection, and freezing."""

from importlib.metadata import version
import math
from pathlib import Path
import platform

from .common import (ENCODER, ENCODER_REVISION, METHODS, PARENTS, SEEDS, RoutingError, bundle, code_hashes, digest, identity,
                     keyed, qkey, read, require, rows, seal, sha, write, write_csv, write_rows)
from .inputs import load_registry
from .pool import pool_questions

CS = (.01, .1, 1., 10.)
THRESHOLDS = (0., .02, .05, .10, .15, .20)


class QuestionEncoder:
    def __init__(self):
        # Optional dependencies are deliberately absent from opportunity/inspect.
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(ENCODER, revision=ENCODER_REVISION, device="cpu")
        self.tokenizer = self.model.tokenizer
        self.limit = self.model.max_seq_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        require(self.limit > 0, "Invalid encoder token window")

    def encode(self, text):
        """Use token windows directly, avoiding decode/re-tokenize truncation."""
        import numpy as np
        import torch
        ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        require(ids, "Empty question token sequence")
        chunks = [ids[i:i + self.limit] for i in range(0, len(ids), self.limit)]
        vectors = []
        self.model.eval()
        with torch.inference_mode():
            for chunk in chunks:
                encoded = self.tokenizer.prepare_for_model(chunk, add_special_tokens=True, truncation=False,
                                                           return_attention_mask=True, return_tensors="pt",
                                                           prepend_batch_axis=True)
                output = self.model(dict(encoded))["sentence_embedding"][0].float().cpu().numpy()
                vectors.append(output)
        mean = np.average(np.asarray(vectors), axis=0, weights=[len(c) for c in chunks])
        norm = np.linalg.norm(mean)
        require(norm > 0 and np.isfinite(mean).all(), "Invalid question embedding")
        return (mean / norm).tolist()


def embed(pool_root, output, encoder=None):
    receipt = bundle(pool_root, "pool")
    root = Path(output)
    if (root / "receipt.json").exists():
        previous = bundle(root, "embeddings")
        require(any(s["path"] == str(Path(pool_root).resolve()) and s["sha256"] == sha(Path(pool_root) / "receipt.json")
                    for s in previous["upstream"]), "Embedding cache belongs to another question pool")
        cached = load_embeddings(root, pool_questions(pool_root))
        return {"questions": len(cached), "dimensions": 384, "cached": True}
    encoder = encoder or QuestionEncoder()
    records = pool_questions(pool_root)
    embedded = []
    for row in records:
        vector = encoder.encode(row["question"])
        require(len(vector) == 384 and all(math.isfinite(v) for v in vector), "Expected finite 384-dimensional embeddings")
        embedded.append({"benchmark": row["benchmark"], "problem_id": row["problem_id"], "vector": vector,
                         "question_sha256": digest(row["question"]),
                         "cache_key": digest([row["question"], ENCODER, ENCODER_REVISION, "token-chunks-weighted-l2-v1"])})
    write_rows(root / "embeddings.jsonl", embedded)
    write(root / "encoder.json", {"model": ENCODER, "revision": ENCODER_REVISION,
          "policy": "token-chunks-weighted-l2-v1", "dimensions": 384,
          "packages": {k: version(k) for k in ("numpy", "torch", "sentence-transformers", "transformers", "tokenizers")} })
    seal(root, "embeddings", [pool_root], receipt["synthetic"])
    return {"questions": len(embedded), "dimensions": 384}


def load_embeddings(root, questions):
    bundle(root, "embeddings")
    config = read(Path(root) / "encoder.json")
    require(config["model"] == ENCODER and config["revision"] == ENCODER_REVISION
            and config["policy"] == "token-chunks-weighted-l2-v1" and config["dimensions"] == 384,
            "Embedding policy changed")
    values = keyed(rows(Path(root) / "embeddings.jsonl"))
    for row in questions:
        item = values[qkey(row)]
        require(item["question_sha256"] == digest(row["question"]) and len(item["vector"]) == 384
                and all(math.isfinite(v) for v in item["vector"]), "Stale/invalid embeddings")
    return {qkey(row): values[qkey(row)]["vector"] for row in questions}


def fit_binary(x, y, c):
    require(x and len(x) == len(y) and all(v in (0, 1) for v in y), "Invalid router labels")
    if len(set(y)) == 1:
        return {"constant": (sum(y) + 1) / (len(y) + 2)}
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=5000,
                                            class_weight=None, random_state=42).fit(x, y)
    except ConvergenceWarning as exc:
        raise RoutingError("Classifier did not converge; do not freeze this experiment") from exc
    return {"coefficient": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0])}


def probability(model, vector):
    if "constant" in model:
        return model["constant"]
    require(len(vector) == len(model["coefficient"]), "Embedding/classifier dimension mismatch")
    score = sum(a * b for a, b in zip(model["coefficient"], vector)) + model["intercept"]
    return 1 / (1 + math.exp(-score)) if score >= 0 else math.exp(score) / (1 + math.exp(score))


def choose(scores, reference, pool, threshold):
    require(reference in pool and set(pool) <= set(scores), "Candidate pool/reference mismatch")
    require(all(math.isfinite(scores[c]) and 0 <= scores[c] <= 1 for c in pool), "Invalid correctness scores")
    alternatives = sorted(c for c in pool if c != reference)
    if not alternatives:
        return reference
    best = min(alternatives, key=lambda c: (-scores[c], c))
    return best if scores[best] - scores[reference] > threshold else reference


def scenarios():
    result = []
    for parent in PARENTS:
        for seed in SEEDS:
            reference = identity(parent, "A", seed)
            policy = [identity(parent, m, seed) for m in METHODS]
            seeds = [identity(parent, "A", s) for s in SEEDS]
            pools = [[reference], [reference, policy[1]], [reference, policy[2]], policy]
            for experiment, candidates, search in (("primary", policy, pools),
                                                   ("policy_three", policy, [policy]),
                                                   ("seed_three", seeds, [seeds])):
                result.append({"id": f"{experiment}_p{parent}_t{seed}", "experiment": experiment,
                               "parent_seed": parent, "continuation_seed": seed, "reference": reference,
                               "candidates": candidates, "pools": search})
    return result


def load_labels(root, split, questions, candidates):
    receipt = bundle(root, "labels")
    metadata = read(Path(root) / "labels.json")
    require(metadata["split"] == split, "Wrong label split")
    result = {}
    for cid in candidates:
        data = keyed(rows(Path(root) / cid / "outcomes.jsonl"))
        require(set(data) == set(keyed(questions)), f"Incomplete label coverage for {cid}")
        result[cid] = {}
        for row in questions:
            item = data[qkey(row)]
            require(type(item["correct"]) is int and item["correct"] in (0, 1)
                    and item["question_sha256"] == digest(row["question"]), "Stale/inconsistent correctness label")
            result[cid][qkey(row)] = item["correct"]
    return result, metadata, receipt


def train(registry_root, pool_root, embeddings_root, train_labels, validation_labels, output):
    registry = load_registry(registry_root)
    for path in (pool_root, embeddings_root, train_labels, validation_labels):
        require(bundle(path)["synthetic"] == registry["synthetic"], "Mixed real and synthetic evidence")
    train_questions = pool_questions(pool_root, ("train",))
    val_questions = pool_questions(pool_root, ("validation",))
    vectors = load_embeddings(embeddings_root, train_questions + val_questions)
    candidate_ids = sorted(registry["candidates"])
    expected = {identity(p, m, s) for p in PARENTS for m in METHODS for s in SEEDS}
    require(set(candidate_ids) == expected, "All 18 declared candidates are required")
    y_train, train_meta, _ = load_labels(train_labels, "train", train_questions, candidate_ids)
    y_val, val_meta, _ = load_labels(validation_labels, "validation", val_questions, candidate_ids)
    for metadata in (train_meta, val_meta):
        require(metadata["checkpoint_hashes"] == {cid: registry["candidates"][cid]["checkpoint_sha256"] for cid in candidate_ids},
                "Labels belong to other checkpoints")
        require(metadata["pool_receipt_sha256"] == sha(Path(pool_root) / "receipt.json"), "Labels belong to another pool")
    fits = {str(c): {cid: fit_binary([vectors[qkey(q)] for q in train_questions],
                                    [y_train[cid][qkey(q)] for q in train_questions], c)
                    for cid in candidate_ids} for c in CS}
    validation_scores = {c: {qkey(q): {cid: probability(model, vectors[qkey(q)]) for cid, model in models.items()}
                             for q in val_questions} for c, models in fits.items()}
    selected, trials = [], []
    for scenario in scenarios():
        options = []
        for c in CS:
            for pool in scenario["pools"]:
                for threshold in THRESHOLDS:
                    choices = [choose(validation_scores[str(c)][qkey(q)], scenario["reference"], pool, threshold) for q in val_questions]
                    correct = sum(y_val[cid][qkey(q)] for q, cid in zip(val_questions, choices))
                    switches = sum(cid != scenario["reference"] for cid in choices)
                    trial = {"scenario": scenario["id"], "c": c, "pool": pool, "threshold": threshold,
                             "validation_accuracy": correct / len(choices), "switches": switches}
                    trials.append(trial)
                    options.append(trial)
        best = min(options, key=lambda o: (-o["validation_accuracy"], o["switches"], len(o["pool"]),
                                          o["c"], tuple(o["pool"]), o["threshold"]))
        fixed = min(scenario["candidates"], key=lambda cid: (-sum(y_val[cid].values()), cid != scenario["reference"], cid))
        selected.append({**scenario, **best, "best_fixed": fixed,
                         "models": {cid: fits[str(best["c"])][cid] for cid in best["pool"]}})
    root = Path(output)
    write(root / "selection.json", selected)
    write_csv(root / "validation_trials.csv", [{**t, "pool": ";".join(t["pool"])} for t in trials])
    write(root / "registry.json", registry)
    write(root / "encoder.json", read(Path(embeddings_root) / "encoder.json"))
    write(root / "environment.json", {"python": platform.python_version(),
          "packages": {k: version(k) for k in ("numpy", "scipy", "scikit-learn")}})
    seal(root, "training", [registry_root, pool_root, embeddings_root, train_labels, validation_labels], registry["synthetic"])
    return {"routers": len(selected), "validation_trials": len(trials), "test_labels_read": False}


def freeze(training_root, output):
    receipt = bundle(training_root, "training")
    require(receipt["code"] == code_hashes(), "Routing code changed after training; create a new experiment")
    source, root = Path(training_root), Path(output)
    selected = read(source / "selection.json")
    require({r["id"] for r in selected} == {r["id"] for r in scenarios()}, "Incomplete router experiment set")
    registry = read(source / "registry.json")
    write(root / "routers.json", selected)
    write(root / "registry.json", registry)
    write(root / "freeze.json", {"training_receipt_sha256": sha(source / "receipt.json"),
          "encoder": ENCODER, "encoder_revision": ENCODER_REVISION, "embedding_policy": "token-chunks-weighted-l2-v1",
          "encoder_environment": read(source / "encoder.json").get("packages", {}),
          "environment": read(source / "environment.json"),
          "purpose": "Frozen before final-test labeling; primary and equal-three diversity controls."})
    seal(root, "frozen", [training_root], receipt["synthetic"])
    return {"routers": len(selected), "frozen_receipt_sha256": sha(root / "receipt.json")}


class FrozenRouter:
    def __init__(self, root, experiment="primary_p17_t42", encoder=None):
        receipt = bundle(root, "frozen")
        require(receipt["code"] == code_hashes(), "Routing code changed after freezing")
        require(not receipt["synthetic"] or encoder is not None, "Synthetic routers cannot be used for live demonstrations")
        config = read(Path(root) / "freeze.json")
        require(config["encoder"] == ENCODER and config["encoder_revision"] == ENCODER_REVISION
                and config["embedding_policy"] == "token-chunks-weighted-l2-v1", "Frozen encoder policy mismatch")
        matches = [r for r in read(Path(root) / "routers.json") if r["id"] == experiment]
        require(len(matches) == 1, "Choose a declared frozen routing experiment")
        self.config = matches[0]
        self.candidates = read(Path(root) / "registry.json")["candidates"]
        if encoder is None:
            require(all(version(k) == v for k, v in config["encoder_environment"].items()),
                    "Embedding runtime differs from the frozen router environment")
        self.encoder = encoder or QuestionEncoder()

    def route(self, text):
        require(isinstance(text, str) and text.strip(), "Question must be nonempty text")
        vector = self.encoder.encode(text)
        scores = {cid: probability(model, vector) for cid, model in self.config["models"].items()}
        selected = choose(scores, self.config["reference"], self.config["pool"], self.config["threshold"])
        return {"selected_checkpoint": selected, "routing_scores": scores,
                "checkpoint_path": self.candidates[selected]["checkpoint"],
                "checkpoint_sha256": self.candidates[selected]["checkpoint_sha256"],
                "threshold": self.config["threshold"], "experiment": self.config["id"]}


def route(question, frozen_root, experiment="primary_p17_t42"):
    return FrozenRouter(frozen_root, experiment).route(question)
