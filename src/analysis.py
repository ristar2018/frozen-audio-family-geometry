# -*- coding: utf-8 -*-
"""
FAPN E1 analysis functions (protocol: Sections 3.4 and 3.5 of the paper)

Confirmatory part (H1a; FolkMusic only, frozen phi, 80% development split):
    - class-level restricted permutation test (instrument class to family surjection), preserving the per-family class counts 6/2/7
    - permutation unit = instrument-class prototype, N = 2000
    - p = (1 + #{sil_perm >= sil_real}) / (N + 1)，Phipson & Smyth (2010)
    - decision: p < 0.01; conjunction over the four encoders (the conjunction is an internal rule of H1a, not a multiple-testing correction)

Descriptive part (fragment-level phi, no p-values):
    - fragment-level silhouette by family, and within/between-family cosine distances

Exploratory part (does not enter the confirmatory decision):
    - linear CKA (label kernel vs embedding kernel, and encoder-to-encoder embedding kernels)

Design constraints (guard against leakage and against spurious significance):
    * permutation runs on the development split only; the test split is isolated by the split script and never enters this module
    * the permutation must be at class level rather than per fragment, so that natural within-class clustering is preserved
      and the null corresponds to "family structure is meaningless"; otherwise any embedding that clusters by instrument would look significant
"""
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
import torch

RANDOM_SEED = 20260501  # H1a permutation seed, fixed before any run (no post-hoc seed switching)


# ============================================================
# Utilities
# ============================================================
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _l2_normalize(emb):
    """L2-normalise row vectors onto the unit sphere (phi and z are both compared by cosine geometry there)."""
    emb = _to_numpy(emb)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm = np.where(norm < 1e-12, 1e-12, norm)
    return emb / norm


def _cosine_silhouette(X, labels):
    """Cosine-distance silhouette; labels are integer or string codes for classes or families."""
    labels = np.asarray(labels)
    # need at least 2 clusters with >=1 member each (singleton clusters score 0, which sklearn handles)
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(silhouette_score(X, labels, metric="cosine"))


# ============================================================
# Class prototypes
# ============================================================
def compute_class_prototypes(embeddings, class_labels, normalize=True):
    """
    Compute each instrument-class prototype as the mean of its fragment embeddings, then L2-normalise.

    Args:
        embeddings: (N, D) fragment embeddings
        class_labels: (N,) instrument-class labels (FolkMusic = instrument_class;
                      NSynth = global entity id "family__entity")
    Returns:
        proto: (K, D) class prototypes (normalised)
        classes: (K,) class label of each prototype (stably sorted)
        counts: (K,) number of fragments per class
    """
    emb = _l2_normalize(embeddings) if normalize else _to_numpy(embeddings)
    class_labels = np.asarray(class_labels)
    classes = np.array(sorted(np.unique(class_labels)))
    protos, counts = [], []
    for c in classes:
        mask = class_labels == c
        proto = emb[mask].mean(axis=0)
        protos.append(proto)
        counts.append(int(mask.sum()))
    proto = np.stack(protos, axis=0)
    if normalize:
        proto = _l2_normalize(proto)
    return proto, classes, np.array(counts)


# ============================================================
# Class-level restricted permutation test (core of H1a)
# ============================================================
def class_level_permutation_test(
    embeddings, class_labels, family_labels,
    n_permutations=2000, seed=RANDOM_SEED,
):
    """
    Class-level restricted permutation test (instrument class to family surjection).

    Steps:
      1) fragment phi -> one prototype per class (K prototypes); true family labels are mapped by class;
      2) observed statistic = silhouette (cosine) of the prototype space grouped by true family;
      3) restricted permutation: keep the number of classes per family fixed, assign the K classes to families at random
         (a surjection onto the families) and recompute the prototype silhouette each time;
      4) p = (1 + #{sil_perm >= sil_real}) / (N + 1)。

    Args:
        embeddings: (N, D) development-split fragment embeddings (phi)
        class_labels: (N,) instrument-class labels
        family_labels: (N,) family labels
        n_permutations: 2000 (fixed in advance)
        seed: random seed (fixed before the run)

    Returns:
        dict: p_value, sil_real, sil_perm, n_classes, family_class_counts, ...
    """
    emb = _l2_normalize(embeddings)
    class_labels = np.asarray(class_labels)
    family_labels = np.asarray(family_labels)

    # class -> true family (a class belongs to exactly one family; take the mode and assert consistency)
    classes = np.array(sorted(np.unique(class_labels)))
    class_to_family = {}
    for c in classes:
        fams = np.unique(family_labels[class_labels == c])
        if len(fams) != 1:
            raise ValueError(f"Class {c!r} maps to multiple families: {fams}")
        class_to_family[c] = fams[0]

    # class prototypes
    proto, proto_classes, _ = compute_class_prototypes(emb, class_labels, normalize=True)
    K = len(proto_classes)
    real_family = np.array([class_to_family[c] for c in proto_classes])

    # family codes (stable sort, for reproducibility)
    fam_names = np.array(sorted(np.unique(real_family)))
    fam_code = {f: i for i, f in enumerate(fam_names)}
    real_family_int = np.array([fam_code[f] for f in real_family])

    # number of classes per family (held fixed under the restricted permutation), in family-code order
    family_class_counts = np.array(
        [int(np.sum(real_family_int == i)) for i in range(len(fam_names))]
    )
    if K != int(family_class_counts.sum()):
        raise AssertionError("family class counts do not sum to n_classes")

    # observed statistic (prototype-space silhouette by family)
    sil_real = _cosine_silhouette(proto, real_family_int)

    # restricted permutation: shuffle class order while keeping the block sizes fixed
    rng = np.random.default_rng(seed)
    base = np.concatenate(
        [np.full(cnt, fi, dtype=int) for fi, cnt in enumerate(family_class_counts)]
    )
    sil_perm = np.empty(n_permutations, dtype=np.float64)
    for b in range(n_permutations):
        perm_idx = rng.permutation(K)          # map the K classes at random onto the fixed family slots
        perm_family = base[perm_idx]
        sil_perm[b] = _cosine_silhouette(proto, perm_family)

    p_value = (1 + int(np.sum(sil_perm >= sil_real))) / (n_permutations + 1)

    return {
        "p_value": float(p_value),
        "sil_real": float(sil_real),
        "sil_perm": sil_perm,
        "sil_perm_mean": float(sil_perm.mean()),
        "sil_perm_std": float(sil_perm.std(ddof=1)),
        "sil_perm_q99": float(np.quantile(sil_perm, 0.99)),
        "n_classes": int(K),
        "n_segments": int(len(emb)),
        "families": list(fam_names),
        "family_class_counts": family_class_counts.tolist(),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "passed": bool(p_value < 0.01),
        "level": "class_prototype",
        "metric": "cosine",
    }


# ============================================================
# Fragment-level descriptive statistics (no confirmatory p-value)
# ============================================================
def segment_level_descriptives(embeddings, family_labels, class_labels=None,
                               max_segments=12000, seed=RANDOM_SEED):
    """
    Descriptive geometry of the fragment-level phi:
      - silhouette by family (fragment level)
      - within- and between-family mean cosine distance, and their ratio
    Large corpora are stratified-sampled down to max_segments to avoid OOM (NSynth has ~10^5 fragments).
    Sampling is stratified by family and the sampling information is recorded in the output.
    """
    emb = _l2_normalize(embeddings)
    family_labels = np.asarray(family_labels)
    n = len(emb)

    sampled = False
    if n > max_segments:
        rng = np.random.default_rng(seed)
        keep = []
        fams = np.unique(family_labels)
        # proportional sampling within each family, keeping at least a few samples
        alloc = np.floor(max_segments * np.array(
            [np.mean(family_labels == f) for f in fams])).astype(int)
        alloc = np.maximum(alloc, 50)
        for f, k in zip(fams, alloc):
            idx = np.where(family_labels == f)[0]
            k = min(k, len(idx))
            keep.append(rng.choice(idx, size=k, replace=False))
        keep = np.concatenate(keep)
        rng.shuffle(keep)
        emb = emb[keep]
        family_labels = family_labels[keep]
        if class_labels is not None:
            class_labels = np.asarray(class_labels)[keep]
        sampled = True

    sil_family = _cosine_silhouette(emb, family_labels)
    dist = family_cosine_distances(emb, family_labels)

    out = {
        "silhouette_family_segment": float(sil_family),
        "d_intra_family": dist["d_intra"],
        "d_inter_family": dist["d_inter"],
        "ratio_inter_over_intra": dist["ratio"],
        "n_segments_used": int(len(emb)),
        "sampled": sampled,
    }
    if class_labels is not None:
        out["silhouette_class_segment"] = float(
            _cosine_silhouette(emb, np.asarray(class_labels)))
    return out


def family_cosine_distances(embeddings, labels):
    """
    Memory-safe within/between-family mean cosine distance (vectorised block processing, no full N x N matrix).
    On the unit sphere the cosine distance is 1 - u.v.
    """
    X = _l2_normalize(embeddings)
    labels = np.asarray(labels)
    uniq = np.array(sorted(np.unique(labels)))
    idx = {u: np.where(labels == u)[0] for u in uniq}

    intra_sum = intra_cnt = 0.0
    inter_sum = inter_cnt = 0.0

    def block_dot(A, B, bs=4096):
        s = 0.0
        for i in range(0, len(A), bs):
            s += float(np.sum(A[i:i + bs] @ B.T))
        return s

    # within family (upper triangle, self-pairs removed)
    for u in uniq:
        ind = idx[u]
        G = X[ind]
        # total number of pairs including self-pairs and both orders: ||G||^2_F
        total = block_dot(G, G)
        diag = float(np.sum(G * G))          # on the unit sphere this equals len(ind)
        # sum of the off-diagonal entries of the upper triangle
        upper_sum = (total - diag) / 2.0
        pairs = len(ind) * (len(ind) - 1) / 2
        if pairs > 0:
            intra_sum += upper_sum
            intra_cnt += pairs

    # between families
    for a in range(len(uniq)):
        for b in range(a + 1, len(uniq)):
            A, B = X[idx[uniq[a]]], X[idx[uniq[b]]]
            s = block_dot(A, B)
            inter_sum += s
            inter_cnt += len(A) * len(B)

    d_intra = 1.0 - intra_sum / intra_cnt if intra_cnt else 0.0
    d_inter = 1.0 - inter_sum / inter_cnt if inter_cnt else 0.0
    ratio = float(d_inter / d_intra) if d_intra > 1e-12 else 0.0
    return {"d_intra": float(d_intra), "d_inter": float(d_inter), "ratio": ratio}


# ============================================================
# CKA (exploratory)
# ============================================================
def linear_cka(X, Y):
    """
    Linear (dot-product) kernel CKA.
    X: (N, D1), Y: (N, D2), same samples in the same order.
    Returns a value in [0, 1]; numerically stable for unit-sphere row vectors.
    """
    X = _to_numpy(X).copy()
    Y = _to_numpy(Y).copy()
    if X.shape[0] != Y.shape[0]:
        raise ValueError("CKA inputs must have the same number of samples")
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # HSIC on Gram matrices; use the inner-product accumulation form to save memory
    K = X @ X.T
    L = Y @ Y.T
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    denom = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(Kc * Lc) / denom)


def onehot_family_kernel(family_labels, classes=None):
    """One-hot matrix of family labels (for the label-kernel vs embedding-kernel CKA)."""
    family_labels = np.asarray(family_labels)
    if classes is None:
        classes = np.array(sorted(np.unique(family_labels)))
    Z = np.zeros((len(family_labels), len(classes)), dtype=np.float64)
    code = {c: i for i, c in enumerate(classes)}
    for i, f in enumerate(family_labels):
        Z[i, code[f]] = 1.0
    return Z


# ============================================================
# Entry points
# ============================================================
def run_e1_folkmusic(embeddings, class_labels, family_labels,
                     n_permutations=2000, seed=RANDOM_SEED, max_segments=12000):
    """
    E1 on FolkMusic: the H1a confirmatory class-level permutation plus fragment-level descriptive statistics.
    The input must be the already-split 80% development set.
    """
    perm = class_level_permutation_test(
        embeddings, class_labels, family_labels,
        n_permutations=n_permutations, seed=seed)
    desc = segment_level_descriptives(
        embeddings, family_labels, class_labels,
        max_segments=max_segments, seed=seed)
    return {"confirmatory": perm, "descriptive": desc}


def run_e1_nsynth_exploratory(embeddings, entity_ids, family_labels,
                              max_segments=12000, seed=RANDOM_SEED):
    """
    E1 on NSynth acoustic: exploratory cross-domain replication (western instrument families), not an H1a decision.
      - class prototypes are instrument entities (family__entity)
      - the same class-level restricted permutation is run, but the result is exploratory evidence only
      - fragment-level descriptive statistics (sampled)
    Families with too few entities (e.g. bass = 1) carry no information for the permutation silhouette
    and should be filtered before the call (data_loader applies min_entities).
    """
    perm = class_level_permutation_test(
        embeddings, entity_ids, family_labels,
        n_permutations=2000, seed=seed)
    perm["confirmatory"] = False   # explicitly marked: does not enter the H1a conjunction
    desc = segment_level_descriptives(
        embeddings, family_labels, entity_ids,
        max_segments=max_segments, seed=seed)
    return {"exploratory": perm, "descriptive": desc}


if __name__ == "__main__":
    # Self-test: toy data with tight classes and separable families; the test should be significant
    rng = np.random.default_rng(0)
    class_centers = {
        "w1": np.array([3, 0.0]), "w2": np.array([2.5, 0.5]),
        "b1": np.array([0.0, 3]), "b2": np.array([0.5, 2.5]),
        "p1": np.array([-3, 0.0]), "p2": np.array([-2.5, -0.5]),
        "p3": np.array([-2.0, 0.4]),
    }
    fam_map = {"w1": "wind", "w2": "wind", "b1": "bowed", "b2": "bowed",
               "p1": "plucked", "p2": "plucked", "p3": "plucked"}
    E, C, F = [], [], []
    for c, cen in class_centers.items():
        for _ in range(60):
            E.append(cen + 0.25 * rng.standard_normal(2))
            C.append(c); F.append(fam_map[c])
    E = np.array(E)
    res = class_level_permutation_test(np.array(E), np.array(C), np.array(F),
                                       n_permutations=500, seed=1)
    print("toy H1a:", {k: res[k] for k in
                       ["sil_real", "p_value", "passed", "family_class_counts",
                        "n_classes", "sil_perm_q99"]})
    d = segment_level_descriptives(np.array(E), np.array(F), np.array(C))
    print("toy descriptives:", d)
