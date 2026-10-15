#!/usr/bin/env python3
"""Score the human-annotated judge-validation set (1-3 author annotators).

Reads the filled annotation CSVs (the `human_genuine` column completed) and
reports, per policy:
  - judge precision   = P(genuine | judge flagged)   [the reward-hacking metric]
  - judge recall proxy = P(judge flagged | genuine)   [uses the judge_label=0 rows]
  - judge-vs-human agreement (Cohen's kappa)
With >=2 annotator files it uses the MAJORITY human label per item and also
reports inter-annotator agreement (mean pairwise Cohen's kappa + Fleiss' kappa).

Usage:
  # preview with the automated heuristic (before humans annotate):
  python scripts/score_annotations.py data/human_eval/annotation_set.csv
  # final, with the three author files:
  python scripts/score_annotations.py data/human_eval/annotation_author{1,2,3}.csv
"""

import csv
import sys
from collections import defaultdict

POLICIES = ("S5", "P1", "B4", "B6")


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def lab(r, col):
    v = (r.get(col) or "").strip()
    return int(v) if v in ("0", "1") else None


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def fleiss_kappa(rows_labels):
    """rows_labels: list of per-item [n0, n1] rater counts (binary categories)."""
    rows = [r for r in rows_labels if sum(r) > 0]
    N = len(rows)
    if N == 0:
        return float("nan")
    n = sum(rows[0])  # raters per item (assume constant)
    if n < 2:
        return float("nan")
    p_j = [sum(r[c] for r in rows) / (N * n) for c in (0, 1)]
    P_i = [(sum(x * x for x in r) - n) / (n * (n - 1)) for r in rows]
    P_bar = sum(P_i) / N
    P_e = sum(p * p for p in p_j)
    return (P_bar - P_e) / (1 - P_e) if P_e != 1 else 1.0


def main():
    paths = sys.argv[1:] or ["data/human_eval/annotation_set.csv"]
    per_file = {p: {r["item_id"]: r for r in load(p)} for p in paths}
    base = load(paths[0])

    # Determine label column: human if any file has human labels, else auto preview.
    any_human = any(
        lab(r, "human_genuine") is not None
        for recs in per_file.values() for r in recs.values()
    )
    human_mode = any_human
    print(f"### {'HUMAN labels' if human_mode else 'AUTO heuristic preview'} "
          f"({len(paths)} file{'s' if len(paths) > 1 else ''})\n")

    # Build the consensus (majority) label per item across annotators.
    def item_label(item_id):
        if not human_mode:
            r = per_file[paths[0]].get(item_id)
            return lab(r, "auto_genuine") if r else None
        votes = [lab(recs[item_id], "human_genuine")
                 for recs in per_file.values() if item_id in recs]
        votes = [v for v in votes if v is not None]
        if not votes:
            return None
        return int(sum(votes) >= (len(votes) / 2.0 + 1e-9))  # majority (ties -> 1)

    by_pol = defaultdict(list)
    for r in base:
        by_pol[r["policy_id"]].append(r["item_id"])

    print(f"{'pol':4s} {'sev':5s} {'flagged':>7s} {'precision':>10s} {'recall':>8s} {'κ(judge,human)':>15s}")
    for pid in POLICIES:
        ids = by_pol.get(pid, [])
        rowmap = {r["item_id"]: r for r in base if r["policy_id"] == pid}
        pos = [i for i in ids if rowmap[i]["judge_label"] == "1"]
        neg = [i for i in ids if rowmap[i]["judge_label"] == "0"]
        ph = [item_label(i) for i in pos]
        ph = [x for x in ph if x is not None]
        precision = sum(ph) / len(ph) if ph else float("nan")
        gp = sum(1 for i in pos if item_label(i) == 1)
        gn = sum(1 for i in neg if item_label(i) == 1)
        recall = gp / (gp + gn) if (gp + gn) else float("nan")
        jl = [int(rowmap[i]["judge_label"]) for i in ids if item_label(i) is not None]
        hl = [item_label(i) for i in ids if item_label(i) is not None]
        k = cohen_kappa(jl, hl) if jl else float("nan")
        sev = "High" if pid in ("S5", "P1") else "Low"
        print(f"{pid:4s} {sev:5s} {len(pos):7d} {precision:9.0%} {recall:7.0%} {k:15.2f}")

    # Inter-annotator agreement + adjudication view (human mode, >=2 files).
    if human_mode and len(paths) >= 2:
        ids = [r["item_id"] for r in base]
        import itertools
        ks = []
        for p, q in itertools.combinations(paths, 2):
            both = [i for i in ids
                    if lab(per_file[p].get(i, {}), "human_genuine") is not None
                    and lab(per_file[q].get(i, {}), "human_genuine") is not None]
            ks.append(cohen_kappa([lab(per_file[p][i], "human_genuine") for i in both],
                                  [lab(per_file[q][i], "human_genuine") for i in both]))
        counts = []
        for i in ids:
            vs = [lab(per_file[p].get(i, {}), "human_genuine") for p in paths]
            vs = [v for v in vs if v is not None]
            if len(vs) == len(paths):
                counts.append([vs.count(0), vs.count(1)])
        extra = f" | Fleiss' κ = {fleiss_kappa(counts):.2f}" if len(paths) >= 3 else ""
        print(f"\nInter-annotator agreement: mean pairwise Cohen's κ = {sum(ks)/len(ks):.2f}{extra}"
              f"  (n={len(counts)} fully-labeled items)")

        # Per-policy: disagreements to adjudicate + strict/lenient precision bounds.
        print(f"\n{'pol':4s} {'flagged':>7s} {'disagree':>9s} {'prec(both=1)':>13s} {'prec(either=1)':>15s}")
        by_pol = defaultdict(list)
        for r in base:
            by_pol[r["policy_id"]].append(r["item_id"])
        n_disagree_total = 0
        for pid in POLICIES:
            rowmap = {r["item_id"]: r for r in base if r["policy_id"] == pid}
            pos = [i for i in by_pol.get(pid, []) if rowmap[i]["judge_label"] == "1"]
            strict = lenient = dis = usable = 0
            for i in pos:
                vs = [lab(per_file[p].get(i, {}), "human_genuine") for p in paths]
                vs = [v for v in vs if v is not None]
                if len(vs) < len(paths):
                    continue
                usable += 1
                if len(set(vs)) > 1:
                    dis += 1
                strict += int(all(v == 1 for v in vs))
                lenient += int(any(v == 1 for v in vs))
            n_disagree_total += dis
            sp = strict / usable if usable else float("nan")
            lp = lenient / usable if usable else float("nan")
            print(f"{pid:4s} {len(pos):7d} {dis:9d} {sp:12.0%} {lp:14.0%}")
        print(f"\n{n_disagree_total} flagged items need adjudication (annotators disagreed). "
              f"Final paper precision = adjudicated labels (between the strict/lenient bounds above).")


if __name__ == "__main__":
    main()
