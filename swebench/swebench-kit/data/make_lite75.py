#!/usr/bin/env python3
"""构建 lite75 = 15 难题(Ornith 稳定不会) + 45 高成功率(双模型都对) + 15 简单题(<15min 且双模型都对).

选题完全确定性(无随机数), 依据三份判分结果投票:
  - Ornith full-500  : mini_v2b_grade  (345/500, mini-corrected 口径)
  - Ornith lite-100  : lite_ornith_grade (48/100, 用于识别"稳定失败" vs 单次翻车)
  - qwen36 full-500  : full_qwen36full_grade (317/500, 交叉投票)
输出: lite75_ids.txt / lite75_filter.txt / lite75_manifest.tsv
"""
import json
from collections import Counter, defaultdict

BASE = "/xttest2/aiden/swe-bench/"

full = [json.loads(l) for l in open(BASE + "verified_test.jsonl")]
by_id = {d["instance_id"]: d for d in full}
lite100 = [l.strip() for l in open(BASE + "lite100_ids.txt") if l.strip()]

orn = json.load(open(BASE + "mini_v2b_grade/ornith-mini-final.mini_v2b.json"))
ornL = json.load(open(BASE + "lite_ornith_grade/lite-ornith.lite_ornith.json"))
qw = json.load(open(BASE + "full_qwen36full_grade/qwen36full.full_qwen36full.json"))
oR, oLR, qR = set(orn["resolved_ids"]), set(ornL["resolved_ids"]), set(qw["resolved_ids"])
oEmpty = set(orn.get("empty_patch_ids", []))

def repo(i): return by_id[i]["repo"]
def diff(i): return by_id[i]["difficulty"]
def plines(i):
    p = by_id[i]["patch"]
    return sum(1 for l in p.splitlines()
               if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---")))

def round_robin(cands, n, taken):
    """按 repo 轮转从已排序候选里取 n 个, 避免单一 repo 扎堆."""
    by_repo = defaultdict(list)
    for i in cands:
        if i not in taken:
            by_repo[repo(i)].append(i)
    order = sorted(by_repo, key=lambda r: (-len(by_repo[r]), r))
    out = []
    while len(out) < n and any(by_repo.values()):
        for r in order:
            if by_repo[r] and len(out) < n:
                out.append(by_repo[r].pop(0))
    return out

taken = set()

# --- A. 15 难题: lite100 里 Ornith(full) 失败的 65 题中挑 ---
fail65 = [i for i in lite100 if i not in oR]
# 排序: 稳定失败(lite 复跑也失败) > 非空 patch(真改错, 不是超时哑火) > qwen 也失败 > patch 大
fail65.sort(key=lambda i: (i in oLR, i in oEmpty, i in qR, -plines(i), i))
hard15 = round_robin(fail65, 15, taken)
taken.update(hard15)

# --- B. 45 高成功率: 双模型都 resolved; 原 35 个 pass 优先(保持历史可比) ---
both = oR & qR
seed = [i for i in lite100 if i in oR and i in both]
seed.sort(key=lambda i: (i not in oLR, i))  # 稳定 pass(lite 复跑也对)排前
seed = seed[:45]
taken.update(seed)
# 补足: lite100 之外的 both-resolved, 优先 15min-1h 档(有内容不算送分)
pool = [i for i in both if i not in taken and i not in set(lite100)]
pool.sort(key=lambda i: (diff(i) != "15 min - 1 hour", plines(i), i))
top_up = round_robin(pool, 45 - len(seed), taken)
high45 = seed + top_up
taken.update(high45)

# --- C. 15 简单题: <15 min fix 且双模型都对, patch 最小优先 ---
easy_pool = [i for i in both if i not in taken and diff(i) == "<15 min fix"]
easy_pool.sort(key=lambda i: (plines(i), i))
easy15 = round_robin(easy_pool, 15, taken)
taken.update(easy15)

cats = [("hard", hard15), ("high", high45), ("easy", easy15)]
assert len(taken) == 75, len(taken)

ids = hard15 + high45 + easy15
with open(BASE + "lite75_ids.txt", "w") as f:
    f.write("\n".join(ids) + "\n")
with open(BASE + "lite75_filter.txt", "w") as f:
    f.write("(" + "|".join(ids) + ")")
with open(BASE + "lite75_manifest.tsv", "w") as f:
    f.write("instance_id\tcategory\trepo\tdifficulty\tpatch_lines\tornith_full\tornith_lite\tqwen36\tin_lite100\n")
    for cat, group in cats:
        for i in group:
            f.write(f"{i}\t{cat}\t{repo(i)}\t{diff(i)}\t{plines(i)}\t"
                    f"{int(i in oR)}\t{int(i in oLR) if i in set(lite100) else ''}\t{int(i in qR)}\t{int(i in set(lite100))}\n")

print("=== lite75 构成 ===")
for cat, group in cats:
    print(f"\n[{cat}] n={len(group)}")
    print("  难度:", dict(Counter(diff(i) for i in group)))
    print("  repo:", dict(Counter(repo(i).split('/')[-1] for i in group)))
    print("  qwen36 resolved:", sum(1 for i in group if i in qR))
    print("  来自原 lite100:", sum(1 for i in group if i in set(lite100)))
print("\n=== 整体 ===")
print("难度:", dict(Counter(diff(i) for i in ids)))
print("Ornith 预期(按 mini_v2b):", sum(1 for i in ids if i in oR), "/75")
print("qwen36 预期:", sum(1 for i in ids if i in qR), "/75")
print("与 lite100 重合:", len(set(ids) & set(lite100)))
print("\n已写出: lite75_ids.txt lite75_filter.txt lite75_manifest.tsv")
