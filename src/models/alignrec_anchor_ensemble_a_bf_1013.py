# models/alignrec_agg.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender

try:
    from .alignrec import ALIGNREC
    from .alignrec_anchor import ALIGNREC_ANCHOR
except Exception:
    from models.alignrec import ALIGNREC
    from models.alignrec_anchor import ALIGNREC_ANCHOR


_MODEL_REGISTRY = {
    "ALIGNREC": ALIGNREC,
    "ALIGNREC_ANCHOR": ALIGNREC_ANCHOR,
}


class ALIGNREC_ANCHOR_ENSEMBLE_A_BF_1013(GeneralRecommender):
    """
    Two-stream(ALIGNREC, ANCHOR) 점수 레벨 앙상블의 '학습형' Bilinear Fusion(+multi-head) 버전.
    최종 점수:
        S = w1 * s1n + w2 * s2n + SUM_h gamma[h] * (s1n_h ⊙ s2n_h)
    - s1n, s2n: 선택적 정규화(zscore/minmax/softmax/none)
    - head별 Hadamard 곱 항의 계수 gamma[h]는 학습 가능
    - w1, w2도 학습 가능
    - 서브모델 파라미터는 freeze
    - calculate_loss: BPR(pairwise)로 (w1, w2, gamma)만 미세학습
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ---- 기본 세팅/체크포인트 ----
        self.device = torch.device(config["device"])

        self.ckpt1 = config['agg_ckpt1'] if 'agg_ckpt1' in config else None
        self.ckpt2 = config['agg_ckpt2'] if 'agg_ckpt2' in config else None
        if not self.ckpt1 or not self.ckpt2:
            raise ValueError("agg_ckpt1, agg_ckpt2 경로를 config에 지정하세요.")

        self.model_name1 = str(config['agg_model1'] if 'agg_model1' in config else "ALIGNREC")
        self.model_name2 = str(config['agg_model2'] if 'agg_model2' in config else "ALIGNREC_ANCHOR")

        # 점수 정규화
        self.norm = str(config['agg_norm'] if 'agg_norm' in config else "zscore").lower()

        # ---- Bilinear Fusion 하이퍼파라미터 ----
        # 아이템 차원을 k개의 head로 균등 chunk
        self.bf_heads = int(config['agg_bf_heads']) if 'agg_bf_heads' in config else 8
        # 초기 gamma 값
        init_gamma = float(config['agg_bf_init_gamma']) if 'agg_bf_init_gamma' in config else 0.1
        # L2 정규화 계수(가벼운 weight decay)
        self.bf_l2 = float(config['agg_bf_l2']) if 'agg_bf_l2' in config else 1e-6

        # ---- Learnable parameters (초기값은 기존 가중합에 맞춤) ----
        init_w1 = float(config['agg_weight1'] if 'agg_weight1' in config else 0.5)
        init_w2 = float(config['agg_weight2'] if 'agg_weight2' in config else 0.5)

        # 선형 결합 계수
        self.w1 = nn.Parameter(torch.tensor(init_w1, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.tensor(init_w2, dtype=torch.float32))
        # head별 gamma (shape: [k])
        self.gamma = nn.Parameter(torch.full((max(1, self.bf_heads),), init_gamma, dtype=torch.float32))

        # ---- 서브모델 로드 & freeze ----
        SubModel1 = _MODEL_REGISTRY[self.model_name1]
        SubModel2 = _MODEL_REGISTRY[self.model_name2]
        self.model1 = SubModel1(config, dataset).to(self.device)
        self.model2 = SubModel2(config, dataset).to(self.device)
        self._safe_load(self.model1, self.ckpt1)
        self._safe_load(self.model2, self.ckpt2)
        self.model1.eval()
        self.model2.eval()
        for p in self.model1.parameters():
            p.requires_grad_(False)
        for p in self.model2.parameters():
            p.requires_grad_(False)

    # ---------- Inference ----------
    @torch.no_grad()
    def full_sort_predict(self, interaction):
        """
        interaction: (user_tensor,) 또는 {'user_id': tensor}
        return: (B, n_items)
        """
        user = self._get_user_tensor(interaction).to(self.device)

        s1 = self.model1.full_sort_predict((user,))  # (B, I)
        s2 = self.model2.full_sort_predict((user,))  # (B, I)

        s1n = self._normalize(s1)
        s2n = self._normalize(s2)

        scores = self._blend_with_bf(s1n, s2n)  # (B, I)
        return scores

    # ---------- Training (learnable BF) ----------
    def calculate_loss(self, interaction, not_train_ui: bool = False):
        """
        BPR(pairwise)로 w1, w2, gamma(head별)만 학습.
        interaction: (users, pos_items, neg_items)
        """
        users = interaction[0].to(self.device)       # (B,)
        pos_items = interaction[1].to(self.device)   # (B,)
        neg_items = interaction[2].to(self.device)   # (B,)

        # 서브모델 점수 (B, I)
        s1 = self.model1.full_sort_predict((users,))
        s2 = self.model2.full_sort_predict((users,))

        s1n = self._normalize(s1)
        s2n = self._normalize(s2)

        scores = self._blend_with_bf(s1n, s2n)  # (B, I)

        # pos/neg 점수 추출
        pos_scores = scores.gather(1, pos_items.view(-1, 1)).squeeze(1)  # (B,)
        neg_scores = scores.gather(1, neg_items.view(-1, 1)).squeeze(1)  # (B,)

        # BPR loss
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

        # 아주 약한 L2 정규화: w1, w2, gamma
        reg = (self.w1 ** 2 + self.w2 ** 2) + (self.gamma ** 2).sum()
        reg = self.bf_l2 * reg

        return bpr + reg

    # ---------- Internal: BF blend ----------
    def _blend_with_bf(self, s1n: torch.Tensor, s2n: torch.Tensor) -> torch.Tensor:
        """
        s1n, s2n: (B, I)
        return: (B, I)
        S = w1*s1n + w2*s2n + Σ_h gamma[h]*(s1n_h ⊙ s2n_h)
        """
        base = self.w1 * s1n + self.w2 * s2n  # (B, I)

        B, I = s1n.shape
        k = max(1, self.bf_heads)
        # 균등 분할
        sizes = [I // k + (1 if x < (I % k) else 0) for x in range(k)]
        offsets = []
        acc = 0
        for sz in sizes:
            offsets.append((acc, acc + sz))
            acc += sz

        out = base
        # head별 Hadamard 곱 항
        for h, (l, r) in enumerate(offsets):
            if l == r:
                continue
            out[:, l:r] = out[:, l:r] + self.gamma[h] * (s1n[:, l:r] * s2n[:, l:r])
        return out

    # ---------- Utils ----------
    def forward(self, *args, **kwargs):
        # 호환용
        return None

    def _get_user_tensor(self, interaction):
        if isinstance(interaction, dict):
            for k in ("user_id", "user", "uid"):
                if k in interaction:
                    return interaction[k]
            raise KeyError("interaction dict에서 user 텐서를 찾을 수 없습니다. (user_id/user/uid 키 확인)")
        else:
            return interaction[0]

    def _normalize(self, s: torch.Tensor) -> torch.Tensor:
        mode = self.norm
        if mode == "none":
            return s
        if mode == "softmax":
            return torch.softmax(s, dim=1)
        if mode == "zscore":
            mu = s.mean(dim=1, keepdim=True)
            sd = s.std(dim=1, keepdim=True).clamp_min(1e-6)
            return (s - mu) / sd
        if mode == "minmax":
            s_min = s.min(dim=1, keepdim=True).values
            s_max = s.max(dim=1, keepdim=True).values
            denom = (s_max - s_min).clamp_min(1e-6)
            return (s - s_min) / denom
        return s

    def _safe_load(self, model: nn.Module, path: str):
        state = torch.load(path, map_location=self.device)
        raw = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        new_state = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if k.startswith("model."):
                    new_state[k[len("model."):]] = v
                elif k.startswith("module."):
                    new_state[k[len("module."):]] = v
                else:
                    new_state[k] = v
        else:
            new_state = raw
        try:
            model.load_state_dict(new_state, strict=True)
        except Exception:
            model.load_state_dict(new_state, strict=False)