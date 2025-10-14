# models/alignrec_agg.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender

# 같은 패키지 내의 모델을 상대 임포트로 시도
try:
    from .alignrec import ALIGNREC
    from .alignrec_anchor import ALIGNREC_ANCHOR
except Exception:
    # 상대 임포트가 환경에 따라 실패할 수 있으므로 절대 임포트 백업
    from models.alignrec import ALIGNREC
    from models.alignrec_anchor import ALIGNREC_ANCHOR


_MODEL_REGISTRY = {
    "ALIGNREC": ALIGNREC,
    "ALIGNREC_ANCHOR": ALIGNREC_ANCHOR,
}


class ALIGNREC_ANCHOR_ENSEMBLE_A_RRF_LEARNABLE_1012(GeneralRecommender):
    """
    두 사전학습 모델을 late-fusion으로 결합.
    - 기본: 학습 가능한 스칼라 가중합 (w1, w2 = softmax(alpha))
    - 옵션: RRF / RankAvg도 그대로 사용 가능(원하면 유지)
    - calculate_loss: BPR(pairwise)로 alpha만 학습 (서브모델 파라미터는 동결)
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 필수 인자
        self.device = torch.device(config['device'])
        self.ckpt1 = config['agg_ckpt1'] if 'agg_ckpt1' in config else None
        self.ckpt2 = config['agg_ckpt2'] if 'agg_ckpt2' in config else None
        if not self.ckpt1 or not self.ckpt2:
            raise ValueError("agg_ckpt1, agg_ckpt2 경로를 config에 지정하세요.")

        self.model_name1 = str(config['agg_model1'] if 'agg_model1' in config else "ALIGNREC")
        self.model_name2 = str(config['agg_model2'] if 'agg_model2' in config else "ALIGNREC_ANCHOR")

        # 점수 정규화 옵션 (score-fusion일 때 사용)
        self.norm = str(config['agg_norm'] if 'agg_norm' in config else "zscore").lower()

        # 앙상블 방식: 'score'(기본) | 'rrf' | 'rankavg'
        self.fusion = str(config['agg_fusion'] if 'agg_fusion' in config else 'score').lower()
        self.rrf_k = int(config['agg_rrf_k'] if 'agg_rrf_k' in config else 75)

        # 학습 가능한 가중치 사용 여부 (기본 True)
        self.learn_weights = bool(config['agg_learn_weights']) if 'agg_learn_weights' in config else True

        # 초기값(선호 모델에 가중치 더 주고 싶으면 여기 값을 조정)
        init_w1 = float(config['agg_weight1']) if 'agg_weight1' in config else 0.5
        init_w2 = float(config['agg_weight2']) if 'agg_weight2' in config else 0.5
        # 로그릿 초기화: softmax([a1,a2]) = [init_w1, init_w2]가 되도록
        eps = 1e-6
        p1 = max(init_w1, eps)
        p2 = max(init_w2, eps)
        a1 = torch.log(torch.tensor(p1)) - torch.log(torch.tensor(p1 + p2))
        a2 = torch.log(torch.tensor(p2)) - torch.log(torch.tensor(p1 + p2))
        self.alpha = nn.Parameter(torch.stack([a1, a2]).float(), requires_grad=self.learn_weights)

        # ---- Sub-models ----
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

        # 임베딩 캐시
        self.cache_embeddings = True
        self._cached = False
        self._u1 = self._i1 = None
        self._u2 = self._i2 = None

    # ---------- Utilities ----------
    def _safe_load(self, model: nn.Module, path: str):
        state = torch.load(path, map_location=self.device)
        raw = state.get("state_dict", state) if isinstance(state, dict) else state
        new_state = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if k.startswith("model."):
                    new_state[k[6:]] = v
                elif k.startswith("module."):
                    new_state[k[7:]] = v
                else:
                    new_state[k] = v
        else:
            new_state = raw
        try:
            model.load_state_dict(new_state, strict=True)
        except Exception:
            model.load_state_dict(new_state, strict=False)

    @torch.no_grad()
    def _get_base_embeddings(self):
        if self.cache_embeddings and self._cached:
            return self._u1, self._i1, self._u2, self._i2
        # 각 모델 forward(norm_adj) -> (user_emb, item_emb, ...)
        u1, i1 = self.model1.forward(self.model1.norm_adj)[:2]
        u2, i2 = self.model2.forward(self.model2.norm_adj)[:2]
        # 필요 시 정규화 없음: 점수 스케일은 학습 가중치가 흡수
        if self.cache_embeddings:
            self._u1, self._i1, self._u2, self._i2 = u1.detach(), i1.detach(), u2.detach(), i2.detach()
            self._cached = True
        return u1, i1, u2, i2

    def _clear_cache(self):
        self._cached = False
        self._u1 = self._i1 = self._u2 = self._i2 = None

    def _weights(self):
        # softmax로 w1,w2 생성
        w = torch.softmax(self.alpha, dim=0)
        return w[0], w[1]

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

    def _get_user_tensor(self, interaction):
        if isinstance(interaction, dict):
            for k in ("user_id", "user", "uid"):
                if k in interaction:
                    return interaction[k]
            raise KeyError("interaction dict에서 user 텐서를 찾을 수 없습니다.")
        else:
            return interaction[0]

    # ---------- Inference ----------
    @torch.no_grad()
    def full_sort_predict(self, interaction):
        user = self._get_user_tensor(interaction).to(self.device)

        # RRF/RankAvg는 그대로 유지 (필요시 사용)
        if self.fusion in ("rrf", "rankavg"):
            s1 = self.model1.full_sort_predict((user,))
            s2 = self.model2.full_sort_predict((user,))
            if self.fusion == "rrf":
                r1 = s1.argsort(dim=1, descending=True).argsort(dim=1) + 1
                r2 = s2.argsort(dim=1, descending=True).argsort(dim=1) + 1
                w1, w2 = self._weights()
                return w1 / (self.rrf_k + r1.float()) + w2 / (self.rrf_k + r2.float())
            else:  # rankavg
                w1, w2 = self._weights()
                r1 = s1.argsort(dim=1, descending=True).argsort(dim=1) + 1
                r2 = s2.argsort(dim=1, descending=True).argsort(dim=1) + 1
                return -(w1 * r1.float() + w2 * r2.float())

        # 기본: score 가중합 (정규화 옵션 적용)
        s1 = self.model1.full_sort_predict((user,))
        s2 = self.model2.full_sort_predict((user,))
        s1n = self._normalize(s1)
        s2n = self._normalize(s2)
        w1, w2 = self._weights()
        return w1 * s1n + w2 * s2n

    # ---------- Training : learnable weights only ----------
    def calculate_loss(self, interaction, not_train_ui: bool = False):
        """
        interaction: (users, pos_items, neg_items)
        - 서브모델 파라미터는 동결, alpha만 업데이트
        - pair-wise BPR on fused score
        """
        if not self.learn_weights:
            # 학습 끌 경우, 0-loss로 통과
            return torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)

        users = interaction[0].to(self.device)
        pos_items = interaction[1].to(self.device)
        neg_items = interaction[2].to(self.device)

        # 캐시된 임베딩으로 pair score 계산
        u1, i1, u2, i2 = self._get_base_embeddings()
        u1_b = u1[users]
        u2_b = u2[users]
        pi1 = i1[pos_items]
        pi2 = i2[pos_items]
        ni1 = i1[neg_items]
        ni2 = i2[neg_items]

        s1_pos = (u1_b * pi1).sum(dim=1)
        s1_neg = (u1_b * ni1).sum(dim=1)
        s2_pos = (u2_b * pi2).sum(dim=1)
        s2_neg = (u2_b * ni2).sum(dim=1)

        w1, w2 = self._weights()
        # (스칼라) w1, w2를 배치에 broadcast
        pos = w1 * s1_pos + w2 * s2_pos
        neg = w1 * s1_neg + w2 * s2_neg

        bpr = -F.logsigmoid(pos - neg).mean()

        # (선택) 작은 L2 규제: alpha의 크기 제한(softmax라 크게 필요 없지만 안정화용)
        reg = 1e-6 * (self.alpha.pow(2).sum())

        return bpr + reg

    def forward(self, *args, **kwargs):
        return None