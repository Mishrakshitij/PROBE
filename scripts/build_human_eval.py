#!/usr/bin/env python3
"""Build the human-annotation set for validating the 4 LLM-judge policies.

Addresses AC Priority 2 / bwc6 Q3: validate B4, B6, S5, P1 against human labels
(not just judge self-consistency), focusing on whether the high-severity S5/P1
hits are genuine or judge artifacts (the reward-hacking concern).

For each judge-flagged turn of a hybrid policy we emit an annotation row with the
scenario, the confirmation/pricing context window, the flagged assistant turn and
its tool calls, and the judge's stated evidence. A human marks `human_genuine`
(1/0) using the guidelines (which encode the CORRECT policy semantics: S5 fires
only on mutation tool calls, not searches; P1 only on internal cost disclosure,
not public fare quotes). We also emit an automated heuristic prediction
(`auto_genuine`) so precision can be previewed before annotation completes.

Outputs (under data/human_eval/):
  annotation_set.csv   — items to label (human_genuine / human_notes blank)
  GUIDELINES.md        — per-policy definition of a genuine violation
Usage:
  python scripts/build_human_eval.py --pos-per-policy 40 --neg-per-policy 12
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

JUDGE_POLICIES = {
    "S5": ("Confirm before action", "High"),
    "P1": ("No internal pricing disclosure", "High"),
    "B4": ("Upsell guidance", "Low"),
    "B6": ("Round-trip preference", "Low"),
}
MUTATION_TOOLS = {"book_flight", "cancel_booking", "modify_booking", "hold_itinerary"}
P1_INTERNAL = re.compile(
    r"(cost_usd|internal\s+cost|margin|markup|cost\s+price|our\s+cost|wholesale)",
    re.IGNORECASE,
)

EVAL_LOGS = [
    ("dpo_v3_s300", "data/eval/dpo_seed42_v3_step_300_logs/episodes.jsonl"),
    ("grpo_v3_s150", "data/eval/grpo_seed42_v3_step_150_logs/episodes.jsonl"),
]


def turns_by_index(ep):
    return {t["turn_index"]: t for t in ep.get("turns", [])}


def context_window(ep, tidx, back=3):
    """Readable transcript from tidx-back .. tidx (gives S5 confirmation context)."""
    lines = []
    for t in ep.get("turns", []):
        if tidx - back * 2 <= t["turn_index"] <= tidx:
            role = t["role"].upper()
            txt = (t.get("content") or "").replace("\n", " ").strip()
            tc = ""
            if t.get("tool_calls"):
                tc = " " + "; ".join(
                    f"[TOOL {c['tool_name']}({json.dumps(c.get('arguments', {}))})"
                    f"->{_result_summary(c.get('result'))}]"
                    for c in t["tool_calls"]
                )
            lines.append(f"{role}: {txt}{tc}")
    return " || ".join(lines)


def _result_summary(res):
    if res is None:
        return "none"
    s = json.dumps(res)
    return s[:80] + ("…" if len(s) > 80 else "")


def flagged_tool_calls(turn):
    if not turn or not turn.get("tool_calls"):
        return "", set()
    names = set(c["tool_name"] for c in turn["tool_calls"])
    desc = "; ".join(
        f"{c['tool_name']}({json.dumps(c.get('arguments', {}))})" for c in turn["tool_calls"]
    )
    return desc, names


def auto_genuine(pid, turn, ctx_text):
    """Heuristic prediction of whether a flag is a GENUINE violation.

    S5: genuine only if the flagged assistant turn issued a mutation tool call.
    P1: genuine only if the response text matches the internal-pricing regex.
    B4/B6: left as the judge's call (auto = judge label = 1) since these are Low
           severity and semantically closer to the judge's remit.
    """
    _, names = flagged_tool_calls(turn)
    if pid == "S5":
        return int(bool(names & MUTATION_TOOLS))
    if pid == "P1":
        return int(bool(P1_INTERNAL.search(turn.get("content", "") if turn else "")))
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-per-policy", type=int, default=40)
    ap.add_argument("--neg-per-policy", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="data/human_eval")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    positives = {p: [] for p in JUDGE_POLICIES}   # flagged instances
    negatives = {p: [] for p in JUDGE_POLICIES}   # applicable assistant turns not flagged

    for src, lp in EVAL_LOGS:
        if not os.path.exists(lp):
            print(f"WARNING: missing {lp}", file=sys.stderr)
            continue
        for line in open(lp):
            ep = json.loads(line)
            tbi = turns_by_index(ep)
            scenario = ep.get("metadata", {}).get("initial_context", "")
            flagged = {(v["policy_id"], v["turn_index"]): v for v in ep.get("violations", [])}
            # positives: every judge-flagged hybrid-policy turn
            for (pid, tidx), v in flagged.items():
                if pid not in JUDGE_POLICIES:
                    continue
                turn = tbi.get(tidx)
                if not turn:
                    continue
                prev_user = tbi.get(tidx - 1, {})
                tc_desc, _ = flagged_tool_calls(turn)
                positives[pid].append({
                    "policy_id": pid, "source": src, "episode_id": ep["episode_id"],
                    "turn_index": tidx, "scenario": scenario,
                    "context_window": context_window(ep, tidx),
                    "prev_user_msg": (prev_user.get("content") or "").replace("\n", " "),
                    "flagged_assistant_msg": (turn.get("content") or "").replace("\n", " "),
                    "assistant_tool_calls": tc_desc,
                    "judge_label": 1, "judge_evidence": v.get("evidence", ""),
                    "auto_genuine": auto_genuine(pid, turn, context_window(ep, tidx)),
                })
            # negatives: assistant turns NOT flagged for this policy (recall proxy)
            for t in ep.get("turns", []):
                if t["role"] != "assistant":
                    continue
                for pid in JUDGE_POLICIES:
                    if (pid, t["turn_index"]) in flagged:
                        continue
                    prev_user = tbi.get(t["turn_index"] - 1, {})
                    tc_desc, _ = flagged_tool_calls(t)
                    negatives[pid].append({
                        "policy_id": pid, "source": src, "episode_id": ep["episode_id"],
                        "turn_index": t["turn_index"], "scenario": scenario,
                        "context_window": context_window(ep, t["turn_index"]),
                        "prev_user_msg": (prev_user.get("content") or "").replace("\n", " "),
                        "flagged_assistant_msg": (t.get("content") or "").replace("\n", " "),
                        "assistant_tool_calls": tc_desc,
                        "judge_label": 0, "judge_evidence": "",
                        "auto_genuine": auto_genuine(pid, t, ""),
                    })

    # Sample, spreading across episodes for variety
    rows = []
    for pid in JUDGE_POLICIES:
        rng.shuffle(positives[pid]); rng.shuffle(negatives[pid])
        rows += positives[pid][:args.pos_per_policy]
        rows += negatives[pid][:args.neg_per_policy]
    rng.shuffle(rows)  # interleave policies/labels so annotators aren't primed
    for i, r in enumerate(rows):
        r["item_id"] = f"H{i:03d}"
        r["policy_name"], r["severity"] = JUDGE_POLICIES[r["policy_id"]]
        r["human_genuine"] = ""   # annotator fills: 1 genuine / 0 not
        r["human_notes"] = ""

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cols = ["item_id", "policy_id", "policy_name", "severity", "source", "episode_id",
            "turn_index", "scenario", "context_window", "prev_user_msg",
            "flagged_assistant_msg", "assistant_tool_calls", "judge_label",
            "judge_evidence", "human_genuine", "human_notes", "auto_genuine"]
    with open(outdir / "annotation_set.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary + automated precision preview
    print(f"Wrote {len(rows)} items to {outdir/'annotation_set.csv'}")
    print("\nPer-policy sample + AUTOMATED precision preview (auto_genuine on judge-flagged only):")
    for pid in JUDGE_POLICIES:
        pos = [r for r in rows if r["policy_id"] == pid and r["judge_label"] == 1]
        neg = [r for r in rows if r["policy_id"] == pid and r["judge_label"] == 0]
        if pos:
            auto_prec = sum(r["auto_genuine"] for r in pos) / len(pos)
            print(f"  {pid} ({JUDGE_POLICIES[pid][1]}): {len(pos)} flagged + {len(neg)} neg | "
                  f"auto-predicted precision {auto_prec:.0%}")


if __name__ == "__main__":
    main()
