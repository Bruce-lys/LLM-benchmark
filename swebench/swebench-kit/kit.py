#!/usr/bin/env python3
"""swebench-kit：一份 YAML 配置驱动整条 SWE-bench 评测管线。

    起服务（vLLM，可选） -> 生成（mini-swe-agent 跑题） -> 评分（官方 swebench harness）

用法（一般通过 ./run.sh 调用）：
    python kit.py CONFIG.yaml [--stages serve,gen,grade] [--dry-run]

所有产物落在 runs/<tag>/ 下：
    vllm.log  mini_swebench.yaml  gen/  gen.log  grade/  kit.log  RESULT.txt
"""
import argparse
import atexit
import copy
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

import yaml

KIT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCKER_PATH = "/usr/local/lib/docker/cli-plugins:/usr/local/bin:/usr/bin:/bin"
OPENHANDS_PY = "/xttest2/aiden/harness/OpenHands/.venv/bin/python"

DEFAULTS = {
    "run": {
        "tag": None,               # 必填。决定输出目录名：runs/<tag>/
        "dataset": "verified",     # verified（全量500）| swelite（自制100题）| lite75（15难+45高成功+15易）| 自备 jsonl 路径
        "split": "test",
        "workers": 64,
        "slice": None,             # 例 "0:5" = 只跑前 5 题
        "filter": None,            # 额外的 instance id 正则，例 "django|sympy"
        "max_passes": 3,           # 自动断点续跑轮数，直到每题都有轨迹
        "retry_empty": True,       # 复测开关：patch 为空的题再生成一遍（口径≈空题上的
                                   # best-of-2，会比单遍略高，对比实验时两边要一致）
                                   # 2026-07-03 起默认开启（Aiden 拍板）；RESULT.txt 会印 on/off
        "output_root": os.path.join(KIT_DIR, "runs"),
    },
    "datasets": {
        # 数据文件随 kit 走（data/ 子目录），部署到哪都不依赖别人的路径
        "verified_jsonl": os.path.join(KIT_DIR, "data", "verified_test.jsonl"),
        "swelite_filter": os.path.join(KIT_DIR, "data", "lite100_filter.txt"),
        "swelite_expected": 100,
        "lite75_filter": os.path.join(KIT_DIR, "data", "lite75_filter.txt"),
        "lite75_expected": 75,
    },
    "model": {
        "path": None,              # 本地权重目录（serve.manage=true 时必填）
        "served_name": None,       # 必填。endpoint 上的模型 id
    },
    "sampling": {
        # temperature/top_p/max_tokens 是 OpenAI 原生参数 -> 直接下发。
        "temperature": 0.85,
        "top_p": 0.95,
        "max_tokens": None,        # 单次回复的输出上限；null = 用服务端默认
        # 非 OpenAI 标准参数走 extra_body 透传给 vLLM（null = 服务端默认）。
        "top_k": None,
        "min_p": None,
        "repetition_penalty": None,
        "extra": {},               # 其他任意请求体参数，例 {seed: 1}
    },
    "agent": {
        "step_limit": 250,
        "cost_limit": None,
    },
    "environment": {
        "timeout": 1800,           # 容器内单条命令超时
        "pull_timeout": 600,
    },
    "serve": {
        "manage": True,            # false = 直接用一个已在运行的 endpoint
        "endpoint": "http://localhost:8000/v1",
        "vllm_bin": "/xttest/grpo/envs/vllm20/bin/vllm",
        "gpus": 2,                 # 整数 N = 自动挑 N 张空闲卡（<2GB 视为空闲，推荐）；
                                   # 也可显式列表如 "6,7"（绕过空闲检查，抢卡自负）
        "tensor_parallel": "auto", # auto = 卡数
        "port": 8000,
        "max_model_len": 262144,   # 上下文窗口
        "gpu_memory_utilization": 0.90,
        "max_num_seqs": 128,
        "max_num_batched_tokens": 65536,
        "tool_call_parser": "qwen3_xml",
        "reasoning_parser": "qwen3",
        "extra_args": [],
        "env": {"VLLM_USE_DEEP_GEMM": "0"},
        "load_timeout": 3600,
        "watchdog": True,          # 服务中途挂掉时自动拉起（第一次全量就死于此）
        "keep_alive": False,       # true = 评测结束后不关服务
    },
    "grade": {
        "enabled": True,
        "rolling": False,          # 滚动评分：生成期间做完一批评一批，评分时间几乎被
                                   # 生成完全遮盖；分数与跑完再评完全一致，只是提前发生
        "rolling_min_batch": 10,   # 攒够多少题评一轮
        "rolling_interval": 120,   # 每隔多少秒检查一次新完成的题
        "dataset_jsonl": "auto",   # auto = 跟随 run.dataset；也可指定 jsonl 路径
        "max_workers": 24,
        "timeout": 1800,
        "namespace": "swebench",   # 从官方仓库拉预构建的逐题镜像
        "python": "auto",          # auto：本 venv 有 swebench 就用它，否则用 OpenHands/.venv
    },
}


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Kit:
    def __init__(self, cfg_path):
        user_cfg = yaml.safe_load(open(cfg_path)) or {}
        self.cfg = deep_merge(DEFAULTS, user_cfg)
        r, m, s = self.cfg["run"], self.cfg["model"], self.cfg["serve"]
        if not r["tag"]:
            sys.exit("配置错误：run.tag 必填")
        if not m["served_name"]:
            sys.exit("配置错误：model.served_name 必填")
        if s["manage"] and not m["path"]:
            sys.exit("配置错误：serve.manage=true 时 model.path 必填")
        self.run_dir = os.path.join(r["output_root"], r["tag"])
        self.gen_dir = os.path.join(self.run_dir, "gen")
        self.grade_dir = os.path.join(self.run_dir, "grade")
        os.makedirs(self.run_dir, exist_ok=True)
        self.logf = open(os.path.join(self.run_dir, "kit.log"), "a")
        self.server_proc = None
        self.dataset = self.resolve_dataset()
        s["gpus"] = self.resolve_gpus()
        if s["tensor_parallel"] in ("auto", None):
            s["tensor_parallel"] = len(str(s["gpus"]).split(","))

    def resolve_dataset(self):
        """把 run.dataset 解析成：mini 的 subset/filter + 期望题数 + 评分用的 jsonl。"""
        r, ds = self.cfg["run"], self.cfg["datasets"]
        name = str(r["dataset"])
        if name == "verified":
            return {"subset": "verified", "filter": None, "expected": 500,
                    "grade_jsonl": ds["verified_jsonl"]}
        if name in ("swelite", "lite75"):
            # swelite/lite75 = verified 的固定子集，靠 instance-id 正则筛选，评分仍用 verified 数据
            # swelite = 100 题（65 Ornith fail + 35 pass，偏难）
            # lite75  = 15 难(Ornith 稳定不会) + 45 高成功率(Ornith/qwen36 双 resolved) + 15 易(<15min)
            if not os.path.exists(ds[f"{name}_filter"]):
                sys.exit(f"{name} 过滤文件不存在：{ds[f'{name}_filter']}")
            if r["filter"]:
                sys.exit(f"配置错误：dataset={name} 时不能再设 run.filter"
                         f"（{name} 本身就是一个 instance-id 过滤器）")
            filt = open(ds[f"{name}_filter"]).read().strip()
            return {"subset": "verified", "filter": filt, "expected": ds[f"{name}_expected"],
                    "grade_jsonl": ds["verified_jsonl"]}
        if os.path.exists(name):  # 自备数据集 jsonl
            try:
                expected = sum(1 for line in open(name) if line.strip())
            except Exception:
                expected = None
            return {"subset": name, "filter": None, "expected": expected, "grade_jsonl": name}
        sys.exit(f"配置错误：run.dataset 只能是 verified | swelite | lite75 | 存在的 jsonl 路径，收到 '{name}'")

    def resolve_gpus(self):
        """serve.gpus 是整数时按显存占用自动挑空闲卡（<2GB 视为空闲）。"""
        g = self.cfg["serve"]["gpus"]
        if not isinstance(g, int):
            return str(g)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"], text=True)
        except Exception:
            sys.exit("serve.gpus 给的是整数（自动挑卡）但 nvidia-smi 不可用——"
                     "请显式列出，例如 gpus: \"4,5,6,7\"")
        stats = []
        for line in out.strip().splitlines():
            idx, mem = [x.strip() for x in line.split(",")]
            stats.append((int(mem), int(idx)))
        idle = sorted(idx for mem, idx in stats if mem < 2048)
        if len(idle) < g:
            sys.exit(f"要 {g} 张空闲卡但只有 {len(idle)} 张空闲（占用<2GB）：{idle}")
        return ",".join(str(i) for i in idle[-g:])

    def log(self, msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    # ---------- 起服务 ----------
    def server_model_id(self):
        try:
            with urllib.request.urlopen(self.cfg["serve"]["endpoint"] + "/models", timeout=5) as f:
                data = json.load(f).get("data") or []
                return data[0]["id"] if data else None
        except Exception:
            return None

    def serve_cmd(self):
        s, m = self.cfg["serve"], self.cfg["model"]
        cmd = [
            s["vllm_bin"], "serve", m["path"],
            "--served-model-name", m["served_name"],
            "--tensor-parallel-size", str(s["tensor_parallel"]),
            "--host", "0.0.0.0", "--port", str(s["port"]),
            "--max-model-len", str(s["max_model_len"]),
            "--gpu-memory-utilization", str(s["gpu_memory_utilization"]),
            "--max-num-seqs", str(s["max_num_seqs"]),
            "--max-num-batched-tokens", str(s["max_num_batched_tokens"]),
            "--enable-chunked-prefill", "--disable-custom-all-reduce",
            "--tool-call-parser", s["tool_call_parser"],
            "--reasoning-parser", s["reasoning_parser"],
            "--enable-auto-tool-choice", "--trust-remote-code",
        ]
        return cmd + list(s["extra_args"])

    def spawn_server(self):
        s = self.cfg["serve"]
        env = os.environ.copy()
        env["PATH"] = os.path.dirname(s["vllm_bin"]) + ":/usr/local/bin:/usr/bin:/bin"
        env["CUDA_VISIBLE_DEVICES"] = str(s["gpus"])
        env.update({k: str(v) for k, v in (s["env"] or {}).items()})
        vllm_log = open(os.path.join(self.run_dir, "vllm.log"), "a")
        self.log("serve: " + shlex.join(self.serve_cmd()))
        self.server_proc = subprocess.Popen(
            self.serve_cmd(), env=env, stdout=vllm_log, stderr=vllm_log,
            stdin=subprocess.DEVNULL, start_new_session=True)
        atexit.register(self.stop_server)

    def stop_server(self):
        if self.server_proc and self.server_proc.poll() is None and not self.cfg["serve"]["keep_alive"]:
            self.log(f"关闭 vLLM（进程组 {self.server_proc.pid}）")
            try:
                os.killpg(self.server_proc.pid, 15)
            except ProcessLookupError:
                pass

    def wait_healthy(self):
        s = self.cfg["serve"]
        deadline = time.time() + s["load_timeout"]
        while time.time() < deadline:
            mid = self.server_model_id()
            if mid:
                self.log(f"endpoint 就绪，服务中：{mid}")
                return
            if self.server_proc and self.server_proc.poll() is not None:
                sys.exit(f"vLLM 提前退出（rc={self.server_proc.returncode}），见 {self.run_dir}/vllm.log")
            time.sleep(15)
        sys.exit("vLLM 在 serve.load_timeout 内没有就绪")

    def ensure_server(self):
        s, m = self.cfg["serve"], self.cfg["model"]
        mid = self.server_model_id()
        if mid:
            if mid != m["served_name"]:
                sys.exit(f"endpoint {s['endpoint']} 正在服务 '{mid}'（配置要的是 '{m['served_name']}'）。"
                         "拒绝动别人的服务——换一个 serve.port 或自行停掉它。")
            self.log(f"复用已在运行的 endpoint（{mid}）")
            return
        if not s["manage"]:
            sys.exit(f"endpoint {s['endpoint']} 不可达，且 serve.manage=false")
        self.spawn_server()
        self.wait_healthy()

    # ---------- 生成 ----------
    def render_mini_config(self):
        """以安装包自带的 swebench.yaml 为模板，套上本次配置，渲染出实际使用的 agent 配置。"""
        import minisweagent
        template = os.path.join(os.path.dirname(minisweagent.__file__),
                                "config", "benchmarks", "swebench.yaml")
        cfg = yaml.safe_load(open(template))
        a, e, sp = self.cfg["agent"], self.cfg["environment"], self.cfg["sampling"]
        cfg.setdefault("agent", {})["step_limit"] = a["step_limit"]
        if a["cost_limit"] is not None:
            cfg["agent"]["cost_limit"] = a["cost_limit"]
        cfg.setdefault("environment", {})["timeout"] = e["timeout"]
        cfg["environment"]["pull_timeout"] = e["pull_timeout"]
        mk = cfg.setdefault("model", {}).setdefault("model_kwargs", {})
        for key in ("temperature", "top_p", "max_tokens"):
            if sp[key] is not None:
                mk[key] = sp[key]
        extra_body = dict(mk.get("extra_body") or {})
        for key in ("top_k", "min_p", "repetition_penalty"):
            if sp[key] is not None:
                extra_body[key] = sp[key]
        extra_body.update(sp["extra"] or {})
        if extra_body:
            mk["extra_body"] = extra_body
        out = os.path.join(self.run_dir, "mini_swebench.yaml")
        yaml.safe_dump(cfg, open(out, "w"), sort_keys=False, allow_unicode=True)
        return out

    def gen_cmd(self, mini_cfg, filter_override=None):
        r, m = self.cfg["run"], self.cfg["model"]
        mini_extra = os.path.join(os.path.dirname(sys.executable), "mini-extra")
        cmd = [mini_extra, "swebench",
               "--subset", self.dataset["subset"], "--split", r["split"],
               "--model", "openai/" + m["served_name"],
               "--workers", str(r["workers"]),
               "-o", self.gen_dir, "-c", mini_cfg]
        if filter_override is not None:  # 复测：只跑指定题目
            return cmd + ["--filter", filter_override]
        if r["slice"]:
            cmd += ["--slice", str(r["slice"])]
        filt = r["filter"] or self.dataset["filter"]
        if filt:
            cmd += ["--filter", str(filt)]
        return cmd

    def gen_env(self):
        env = os.environ.copy()
        env["PATH"] = os.path.dirname(sys.executable) + ":" + DOCKER_PATH
        env["OPENAI_API_BASE"] = self.cfg["serve"]["endpoint"]
        env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "dummy")
        env["MSWEA_COST_TRACKING"] = "ignore_errors"
        env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        return env

    def traj_count(self):
        return len(glob.glob(os.path.join(self.gen_dir, "*", "*.traj.json")))

    def empty_patch_ids(self):
        """列出轨迹存在但 patch 为空（或轨迹损坏）的题目 id。"""
        ids = []
        for f in glob.glob(os.path.join(self.gen_dir, "*", "*.traj.json")):
            iid = os.path.basename(f)[: -len(".traj.json")]
            try:
                traj = json.load(open(f))
            except Exception:
                ids.append(iid)
                continue
            if not ((traj.get("info") or {}).get("submission") or "").strip():
                ids.append(iid)
        return sorted(ids)

    def retry_empty_pass(self, mini_cfg, env, gen_log):
        """复测：把空 patch 题的旧轨迹挪到备份目录，再针对性重新生成一遍。"""
        empties = self.empty_patch_ids()
        if not empties:
            self.log("复测：没有空 patch 的题，跳过")
            return
        backup = os.path.join(self.run_dir, "gen_empty_pass1")
        os.makedirs(backup, exist_ok=True)
        for iid in empties:
            src = os.path.join(self.gen_dir, iid)
            if os.path.isdir(src):
                dst = os.path.join(backup, iid)
                shutil.rmtree(dst, ignore_errors=True)
                shutil.move(src, dst)
        self.log(f"复测：{len(empties)} 题 patch 为空，旧轨迹已备份到 {backup}，开始重新生成")
        filt = "(" + "|".join(re.escape(i) for i in empties) + ")"
        rc = subprocess.call(self.gen_cmd(mini_cfg, filter_override=filt), env=env,
                             stdout=gen_log, stderr=gen_log, stdin=subprocess.DEVNULL)
        still = self.empty_patch_ids()
        self.log(f"复测完成 rc={rc}：{len(empties) - len(still)}/{len(empties)} 题拿到了非空 patch，"
                 f"仍为空 {len(still)} 题")

    def watchdog_loop(self, stop_evt):
        """生成期间每分钟探活；服务挂了且归本 kit 管，就地拉起。"""
        while not stop_evt.wait(60):
            if self.server_model_id():
                continue
            self.log("watchdog：endpoint 掉线")
            if not (self.cfg["serve"]["manage"] and self.cfg["serve"]["watchdog"]):
                continue
            if self.server_proc and self.server_proc.poll() is None:
                try:
                    os.killpg(self.server_proc.pid, 9)
                except ProcessLookupError:
                    pass
            self.log("watchdog：重启 vLLM")
            self.spawn_server()
            self.wait_healthy()

    def generate(self, mini_cfg):
        r = self.cfg["run"]
        # 设了 slice/filter 时无法预知题数，走"跑通即止"；否则按数据集期望题数断点续跑
        expected = None if (r["slice"] or r["filter"]) else self.dataset["expected"]
        cmd, env = self.gen_cmd(mini_cfg), self.gen_env()
        self.log("generate: " + shlex.join(cmd))
        stop_evt = threading.Event()
        wd = threading.Thread(target=self.watchdog_loop, args=(stop_evt,), daemon=True)
        wd.start()
        gen_log = open(os.path.join(self.run_dir, "gen.log"), "a")
        try:
            for p in range(1, r["max_passes"] + 1):
                self.log(f"生成第 {p}/{r['max_passes']} 轮 "
                         f"(temp={self.cfg['sampling']['temperature']} top_p={self.cfg['sampling']['top_p']} "
                         f"steps={self.cfg['agent']['step_limit']} workers={r['workers']})")
                rc = subprocess.call(cmd, env=env, stdout=gen_log, stderr=gen_log,
                                     stdin=subprocess.DEVNULL)
                n = self.traj_count()
                self.log(f"第 {p} 轮结束 rc={rc}，累计轨迹：{n}" +
                         (f"/{expected}" if expected else ""))
                if expected is None:
                    if rc == 0 or p >= 2:
                        break
                elif n >= expected:
                    break
            if r["retry_empty"]:
                self.retry_empty_pass(mini_cfg, env, gen_log)
        finally:
            stop_evt.set()
        self.log(f"生成完成：{self.traj_count()} 条轨迹，位于 {self.gen_dir}")

    # ---------- 评分 ----------
    def grader_python(self):
        want = self.cfg["grade"]["python"]
        if want != "auto":
            return want
        try:
            import swebench  # noqa: F401
            return sys.executable
        except ImportError:
            if os.path.exists(OPENHANDS_PY):
                return OPENHANDS_PY
            sys.exit("找不到 swebench 包：先 `./run.sh --setup`，或手动指定 grade.python")

    @staticmethod
    def touches_tests(patch):
        """patch 是否改动了测试类文件（tests/ 目录、test_*.py、conftest.py 等）。
        官方评分会重置 gold 测试补丁涉及的文件，但 conftest 等测试基础设施不在
        重置范围内 —— 碰了这些文件的题存在"改题"嫌疑，需要人工复核。"""
        for m in re.finditer(r"^diff --git a/(\S+) b/", patch, re.M):
            p = "/" + m.group(1).lower()
            base = os.path.basename(p)
            if ("/tests/" in p or "/testing/" in p or base == "conftest.py"
                    or base.startswith("test_") or base.endswith("_test.py")):
                return True
        return False

    def graded_ids(self):
        """已出过逐题报告（评过分）的题目 id 集合。"""
        tag = self.cfg["run"]["tag"]
        return {os.path.basename(os.path.dirname(f))
                for f in glob.glob(os.path.join(self.grade_dir, "logs", "run_evaluation",
                                                tag, "*", "*", "report.json"))}

    def collect_preds(self, skip_empty=False, exclude_graded=False):
        """从轨迹里抽出每题的 patch。skip_empty 跳过空 patch；exclude_graded 跳过已评的题。"""
        tag = self.cfg["run"]["tag"]
        graded = self.graded_ids() if exclude_graded else set()
        preds = {}
        for f in glob.glob(os.path.join(self.gen_dir, "*", "*.traj.json")):
            iid = os.path.basename(f)[:-len(".traj.json")]
            if iid in graded:
                continue
            try:
                traj = json.load(open(f))
            except Exception:
                continue
            patch = (traj.get("info") or {}).get("submission") or ""
            if skip_empty and not patch.strip():
                continue
            preds[iid] = {"instance_id": iid, "model_name_or_path": tag, "model_patch": patch}
        return preds

    def write_preds(self, preds):
        path = os.path.join(self.grade_dir, "preds.json")
        json.dump(preds, open(path, "w"))
        return path

    def aggregate_resolved(self):
        """从逐题报告聚合 resolved 数（滚动评分的进度和最终报告都用它）。"""
        tag = self.cfg["run"]["tag"]
        resolved = 0
        for f in glob.glob(os.path.join(self.grade_dir, "logs", "run_evaluation",
                                        tag, "*", "*", "report.json")):
            try:
                d = json.load(open(f))
                resolved += sum(1 for v in d.values() if v.get("resolved"))
            except Exception:
                continue
        return resolved

    def _run_evaluation(self, preds_path):
        """跑一轮官方评分器（含容器预清理和网络补丁）。"""
        g, r = self.cfg["grade"], self.cfg["run"]
        py = self.grader_python()
        dataset = self.dataset["grade_jsonl"] if g["dataset_jsonl"] == "auto" else g["dataset_jsonl"]
        args = ["--dataset_name", dataset, "--split", r["split"],
                "--predictions_path", preds_path, "--run_id", r["tag"],
                "--max_workers", str(g["max_workers"]),
                "--cache_level", "instance", "--namespace", g["namespace"],
                "--timeout", str(g["timeout"])]
        # swebench 4.0.4 两个已知网络坑（国内直连 GitHub 会永久挂起）：
        # 1) make_run_report 传了 docker client 就会为全量数据集重建 test spec，
        #    逐题无超时地拉 raw.githubusercontent 的 requirements —— 把 client 拍成
        #    None 跳过（只损失"未清理镜像/容器"的装饰性统计，分数不受影响）；
        # 2) 其余 requests.get 没有 timeout —— 用 socket 全局超时兜底。
        wrapper = (
            "import socket, runpy, sys; socket.setdefaulttimeout(60)\n"
            "import swebench.harness.reporting as _rep\n"
            "_orig = _rep.make_run_report\n"
            "_rep.make_run_report = lambda preds, ds, rid, client=None: _orig(preds, ds, rid, None)\n"
            f"sys.argv = ['run_evaluation'] + {args!r}\n"
            "runpy.run_module('swebench.harness.run_evaluation', run_name='__main__')")
        env = os.environ.copy()
        env["PATH"] = DOCKER_PATH
        grade_log = open(os.path.join(self.grade_dir, "grade.log"), "a")
        # 上一次评分若被中断，会留下同名 eval 容器（sweb.eval.<iid>.<tag>），
        # 重评时 docker 409 冲突 → 该题被记为 error。先清掉再跑。
        # 匹配收紧到 sweb.eval. 前缀 + 本 run 的 tag 结尾，绝不会碰其他人的容器；
        # 评分轮次严格串行（滚动线程与最终扫尾不重叠），不会误清在跑的容器。
        subprocess.call(["bash", "-c",
                         f"docker ps -aq --filter 'name=^sweb\\.eval\\..*\\.{r['tag']}$' | xargs -r docker rm -f"],
                        env=env, stdout=grade_log, stderr=grade_log, stdin=subprocess.DEVNULL)
        self.log("grade: swebench.harness.run_evaluation " + shlex.join(args))
        subprocess.call([py, "-c", wrapper], cwd=self.grade_dir, env=env,
                        stdout=grade_log, stderr=grade_log, stdin=subprocess.DEVNULL)

    def rolling_grade_loop(self, stop_evt):
        """滚动评分：生成期间每攒够一批新完成的题就评一轮。
        只评非空 patch（空题留给复测/最终扫尾）；评分失败无报告的题下轮自动重试。"""
        g = self.cfg["grade"]
        os.makedirs(self.grade_dir, exist_ok=True)
        while not stop_evt.wait(g["rolling_interval"]):
            preds = self.collect_preds(skip_empty=True, exclude_graded=True)
            if len(preds) < g["rolling_min_batch"]:
                continue
            self.log(f"滚动评分：{len(preds)} 题新完成，开始本轮评分")
            self._run_evaluation(self.write_preds(preds))
            self.log(f"滚动评分：累计已评 {len(self.graded_ids())} 题，"
                     f"当前 resolved {self.aggregate_resolved()}")

    def grade(self):
        """最终评分/扫尾：滚动模式下只评剩余未评的题，否则全量评一轮。"""
        g = self.cfg["grade"]
        os.makedirs(self.grade_dir, exist_ok=True)
        all_preds = self.collect_preds()
        patched = sum(1 for v in all_preds.values() if v["model_patch"].strip())
        self.log(f"preds：共 {len(all_preds)} 题，其中 {patched} 题有非空 patch")
        # 改题嫌疑检测：patch 碰了测试类文件的题单独列出
        suspects = sorted(i for i, v in all_preds.items() if self.touches_tests(v["model_patch"]))
        if suspects:
            with open(os.path.join(self.grade_dir, "touches_tests.txt"), "w") as fh:
                fh.write("\n".join(suspects) + "\n")
            self.log(f"注意：{len(suspects)} 题的 patch 改动了测试类文件，已列入 "
                     f"grade/touches_tests.txt —— 官方评分会重置 gold 测试文件，"
                     f"但这些题建议人工复核是否借改测试环境过关")
        preds = self.collect_preds(exclude_graded=True) if g["rolling"] else all_preds
        if preds:
            self._run_evaluation(self.write_preds(preds))
        else:
            self.log("所有题都已在滚动评分中评完，直接汇总")
        return self.report()

    def report(self):
        r, g = self.cfg["run"], self.cfg["grade"]
        reports = [f for f in glob.glob(os.path.join(self.grade_dir, "*.json"))
                   if not f.endswith("preds.json")]
        # 滚动模式下每轮的总报告都只覆盖该轮的题，必须从逐题报告聚合
        if reports and not g["rolling"]:
            rep = json.load(open(max(reports, key=os.path.getmtime)))
            resolved = rep.get("resolved_instances", 0)
            total = rep.get("total_instances") or rep.get("submitted_instances") or 0
            # 跑的是子集时分母不能用 swebench 报告的 total_instances——那个恒等于
            # grade_jsonl 的全量（swelite/lite75 评分仍用 verified → 恒 500）
            if (r["slice"] or r["filter"]) and rep.get("submitted_instances"):
                total = rep["submitted_instances"]  # slice/额外filter 优先：分母 = 实际提交数
            elif self.dataset["filter"]:  # swelite / lite75 固定子集：分母 = 子集题数
                total = self.dataset["expected"] or rep.get("submitted_instances") or total
        else:
            # 从逐题 report.json 聚合（滚动模式的正路；单轮模式总报告缺失时的兜底）
            graded = self.graded_ids()
            if graded:
                resolved = self.aggregate_resolved()
                if r["slice"] or r["filter"]:  # 跑的是子集：分母用实际生成的题数
                    total = self.traj_count() or len(graded)
                else:
                    total = self.dataset["expected"] or len(graded)
                self.log(f"已从 {len(graded)} 份逐题报告聚合")
            elif reports:
                # 一份逐题报告都没有（例如所有实例 patch 无法 apply 被评分器记为
                # error，不产出 report.json）——退回总报告口径，error 按未解决计入
                rep = json.load(open(max(reports, key=os.path.getmtime)))
                resolved = rep.get("resolved_instances", 0)
                total = rep.get("submitted_instances") or self.traj_count() or 0
                if self.dataset["filter"] and not (r["slice"] or r["filter"]):
                    total = self.dataset["expected"] or total
                self.log("警告：无逐题报告（可能所有实例评分出错），已按总报告聚合")
            else:
                self.log("警告：没找到任何评分报告，请查看 grade/grade.log")
                return None
        pct = round(100 * resolved / total, 1) if total else 0
        engine = ""  # 推理引擎版本戳：vllm_bin 指向共享环境、不受 uv.lock 锁定，
        try:         # 版本漂移会影响分数（0.20→0.22 实测有感），必须印在结果里
            m = re.search(r"LLM engine \((v[0-9][^)]*)\)",
                          open(os.path.join(self.run_dir, "vllm.log")).read())
            if m:
                engine = f" engine={m.group(1)}"
        except Exception:
            pass  # manage:false 连外部服务时无 vllm.log，不戳
        summary = (f"tag={r['tag']} model={self.cfg['model']['served_name']} dataset={r['dataset']} "
                   f"temp={self.cfg['sampling']['temperature']} top_p={self.cfg['sampling']['top_p']} "
                   f"steps={self.cfg['agent']['step_limit']} "
                   f"retry_empty={'on' if r['retry_empty'] else 'off'}{engine}\n"
                   f"RESOLVED {resolved}/{total} = {pct}%\n"
                   f"gen: {self.gen_dir}\ngrade: {self.grade_dir}\n")
        open(os.path.join(self.run_dir, "RESULT.txt"), "w").write(summary)
        self.log("FINAL " + summary.replace("\n", " | "))
        return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--stages", default="serve,gen,grade",
                    help="逗号分隔的阶段子集：serve,gen,grade")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stages = set(args.stages.split(","))

    kit = Kit(args.config)
    kit.log(f"配置：{os.path.abspath(args.config)} -> 输出目录 {kit.run_dir}")
    if args.dry_run:
        mini_cfg = kit.render_mini_config()
        kit.log("解析后的完整配置：\n" + yaml.safe_dump(kit.cfg, sort_keys=False, allow_unicode=True))
        kit.log("serve 命令: " + shlex.join(kit.serve_cmd()))
        kit.log("gen 命令:   " + shlex.join(kit.gen_cmd(mini_cfg)))
        kit.log(f"渲染出的 mini 配置：{mini_cfg}")
        kit.log("dry run —— 未执行任何操作")
        return

    if "serve" in stages or "gen" in stages:
        kit.ensure_server()
    if "gen" in stages:
        # 滚动评分：生成期间由后台线程做完一批评一批；线程与最终扫尾严格串行
        rolling = (kit.cfg["grade"]["rolling"] and kit.cfg["grade"]["enabled"]
                   and "grade" in stages)
        rg_stop, rg_thread = threading.Event(), None
        if rolling:
            kit.log(f"滚动评分已开启：每攒够 {kit.cfg['grade']['rolling_min_batch']} 题评一轮"
                    f"（每 {kit.cfg['grade']['rolling_interval']}s 检查）")
            rg_thread = threading.Thread(target=kit.rolling_grade_loop,
                                         args=(rg_stop,), daemon=True)
            rg_thread.start()
        kit.generate(kit.render_mini_config())
        if rg_thread:
            rg_stop.set()
            rg_thread.join()  # 等在跑的一轮评完，避免与最终扫尾并发
    if "grade" in stages and kit.cfg["grade"]["enabled"]:
        kit.grade()
    kit.stop_server()


if __name__ == "__main__":
    main()
