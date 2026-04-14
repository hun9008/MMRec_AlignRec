import re
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt


# =============================
# Robust log parsing (state machine)
# =============================
_PARAMS_RE = re.compile(
    r"Parameters:\s*\[(?P<names>[^\]]+)\]\s*=\s*\((?P<values>[^)]*)\)",
    re.IGNORECASE,
)

_BEST_RE = re.compile(r"\bbest\s+(valid|test)\s*:\s*(?P<body>.*)$", re.IGNORECASE)


def _split_csv_like(s: str) -> List[str]:
    # "'seed', 'lambda_weight', ..." 형태 안전 분리
    return [x.strip().strip("'").strip('"') for x in s.split(",") if x.strip()]


def _parse_tuple_values(s: str) -> List[Any]:
    # "(333, 0.01, 0.0001, ...)" 안의 값들을 int/float로 파싱
    raw = [x.strip() for x in s.split(",") if x.strip()]
    vals: List[Any] = []
    for x in raw:
        if re.fullmatch(r"-?\d+", x):
            vals.append(int(x))
        else:
            # 1e-4 같은 과학표기도 float로 처리
            vals.append(float(x))
    return vals


def _extract_metric(body: str, metric_name: str) -> Optional[float]:
    # metric 예: ndcg@20, recall@10
    metric_re = re.compile(rf"\b{re.escape(metric_name)}\s*:\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE)
    m = metric_re.search(body)
    if not m:
        return None
    return float(m.group(1))


def parse_log_for_param_metric(
    log_path: str,
    param_name: str,
    metric_name: str,
    split: str = "valid",  # "valid" or "test"
) -> List[Tuple[float, float]]:
    """
    returns: list of (param_value, metric_value)
    - Parameters 라인과 best valid/test 라인이 여러 줄로 분리돼도 파싱됨
    """
    split = split.lower()
    pairs: List[Tuple[float, float]] = []

    current_params: Optional[Dict[str, Any]] = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 1) Parameters 라인 갱신
            if "Parameters:" in line:
                m = _PARAMS_RE.search(line)
                if not m:
                    current_params = None
                    continue

                names = _split_csv_like(m.group("names"))
                values = _parse_tuple_values(m.group("values"))

                if len(names) != len(values):
                    current_params = None
                    continue

                current_params = dict(zip(names, values))
                continue

            # 2) best valid/test 라인 파싱 (Parameters 이후에 나오는 걸 current_params로 연결)
            if current_params is None:
                continue

            mb = _BEST_RE.search(line)
            if not mb:
                continue

            section = mb.group(1).lower()
            if section != split:
                continue

            body = mb.group("body")
            mv = _extract_metric(body, metric_name)
            if mv is None:
                continue

            if param_name not in current_params:
                continue

            pv = current_params[param_name]
            # 숫자로 캐스팅 가능한 경우만 처리
            try:
                pv_f = float(pv)
            except Exception:
                continue

            pairs.append((pv_f, float(mv)))

    return pairs


# =============================
# Plot (paper-like hyperparam curve)
# =============================
def plot_param_mean_metric_paperlike(
    log_path: str,
    param_name: str,
    metric_name: str,
    split: str = "test",
    x_label: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    log_x: bool = True,
    x_equal_spacing: bool = True,
    annotate_n: bool = False,
    figsize: Tuple[float, float] = (3.2, 2.2),  # 논문 small figure 느낌
):
    pairs = parse_log_for_param_metric(
        log_path=log_path,
        param_name=param_name,
        metric_name=metric_name,
        split=split,
    )

    if not pairs:
        raise RuntimeError(
            f"No data parsed. Check param='{param_name}', metric='{metric_name}', split='{split}', file='{log_path}'."
        )

    bucket = defaultdict(list)
    for p, v in pairs:
        bucket[p].append(v)

    xs = sorted(bucket.keys())
    ys = [float(np.mean(bucket[x])) for x in xs]
    ns = [len(bucket[x]) for x in xs]

    # ---- 스타일: 두번째 그림처럼 (얇은 축, 작은 폰트, 파란 빈 사각형)
    plt.figure(figsize=figsize, dpi=200)

    ax = plt.gca()
    if x_equal_spacing:
        x_pos = list(range(len(xs)))
    else:
        x_pos = xs

    ax.plot(
        x_pos,
        ys,
        marker="s",
        linestyle="-",
        linewidth=1.0,
        markersize=4.0,
        color="#2f5aff",
        markerfacecolor="white",
        markeredgecolor="#2f5aff",
        markeredgewidth=1.0,
    )

    # x축: 값 간격을 동일하게 보여주기 (스크린샷 스타일)
    if x_equal_spacing:
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{v:g}" for v in xs], rotation=30, ha="right")
        ax.set_xlim(-0.4, len(xs) - 0.6)
    else:
        # log x축 (0.0001~1 같이 스케일 차이 큰 경우 논문에서 흔히 log)
        if log_x:
            ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, pos: f"{v:g}"))

    # grid는 거의 안 보이게 / 또는 아예 끄기
    ax.grid(False)

    # 축/틱 얇게 + 폰트 작게
    ax.tick_params(axis="both", which="both", labelsize=8, width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    if x_label is None:
        x_label = param_name
    ax.set_xlabel(x_label, fontsize=14, weight='bold')
    ax.set_ylabel(metric_name.upper(), fontsize=9, weight='bold')


    # y 범위 여백 조금만
    y_min, y_max = min(ys), max(ys)
    margin = (y_max - y_min) * 0.15 if y_max > y_min else 0.01
    ax.set_ylim(y_min - margin, y_max + margin)

    if annotate_n:
        for x, y, n in zip(x_pos, ys, ns):
            ax.text(x, y, f"{n}", fontsize=7, va="bottom", ha="left")

    plt.tight_layout(pad=0.5)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return xs, ys, ns


# =============================
# Example
# =============================
if __name__ == "__main__":

    data = "elec"

    plot_param_mean_metric_paperlike(
        log_path=f"./log/ANCHORREC-{data}-hyper-tau.log",
        param_name="tau_weight",
        metric_name="recall@20",
        split="test",
        save_path=f"./hyper/{data}_tau_weight_recall20_test.png",
        show=False,
        log_x=False,                     # x_equal_spacing=True면 log_x는 무시됨
        x_equal_spacing=True,
        annotate_n=False,                # 두번째 그림은 n 표기 거의 없음
        figsize=(3.2, 2.2),
        x_label = r"$\tau$"
    )

    plot_param_mean_metric_paperlike(
        log_path=f"./log/ANCHORREC-{data}-hyper-tau.log",
        param_name="tau_weight",
        metric_name="ndcg@20",
        split="test",
        save_path=f"./hyper/{data}_tau_weight_ndcg20_test.png",
        show=False,
        log_x=False,                     # x_equal_spacing=True면 log_x는 무시됨
        x_equal_spacing=True,
        annotate_n=False,                # 두번째 그림은 n 표기 거의 없음
        figsize=(3.2, 2.2),
        x_label = r"$\tau$"
    )