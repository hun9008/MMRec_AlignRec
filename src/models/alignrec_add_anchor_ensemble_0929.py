# models/alignrec_agg.py
import torch
import torch.nn as nn

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


class ALIGNREC_ADD_ANCHOR_ENSEMBLE_0929(GeneralRecommender):
    """
    독립적으로 학습된 두 모델을 로드하여 추론 시 최종 점수만 집계하는 앙상블 모델.
    - 학습은 alignrec.py / alignrec_anchor.py 등 각자에서 수행
    - 본 클래스는 추론 전용이지만, MMRec 트레이너의 인터페이스를 깨지 않도록
      calculate_loss()에서 '0-loss'를 반환하여 학습 루프가 안전하게 통과되도록 한다.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # Config 객체는 dict처럼 인덱싱
        self.ckpt1 = config['agg_ckpt1'] if 'agg_ckpt1' in config else None
        self.ckpt2 = config['agg_ckpt2'] if 'agg_ckpt2' in config else None
        if not self.ckpt1 or not self.ckpt2:
            raise ValueError("agg_ckpt1, agg_ckpt2 경로를 config에 지정하세요.")

        self.model_name1 = str(config['agg_model1'] if 'agg_model1' in config else "ALIGNREC")
        self.model_name2 = str(config['agg_model2'] if 'agg_model2' in config else "ALIGNREC_ANCHOR")
        # ...
        self.w1 = float(config['agg_weight1'] if 'agg_weight1' in config else 0.5)
        self.w2 = float(config['agg_weight2'] if 'agg_weight2' in config else 0.5)
        self.norm = str(config['agg_norm'] if 'agg_norm' in config else "zscore").lower()
        self.device = torch.device(config['device'])
        
        # ---- Sub-models ----
        # 동일한 dataset을 전달하여 user/item index space를 일치시킨다.
        SubModel1 = _MODEL_REGISTRY[self.model_name1]
        SubModel2 = _MODEL_REGISTRY[self.model_name2]

        self.model1 = SubModel1(config, dataset).to(self.device)
        self.model2 = SubModel2(config, dataset).to(self.device)

        self._safe_load(self.model1, self.ckpt1)
        self._safe_load(self.model2, self.ckpt2)

        self.model1.eval()
        self.model2.eval()

        # 학습시 파라미터 업데이트가 일어나지 않도록 보장
        for p in self.model1.parameters():
            p.requires_grad_(False)
        for p in self.model2.parameters():
            p.requires_grad_(False)

        # 트레이너가 optimizer를 생성할 때 빈 파라미터 리스트여도 PyTorch는 허용하므로 추가 조치는 필요 없음.

    # ---------- Inference ----------
    @torch.no_grad()
    def full_sort_predict(self, interaction):
        """
        interaction: (user_tensor, ) 또는 {'user_id': tensor} 형태 모두 허용
        return: (B, n_items) 점수 텐서
        """
        user = self._get_user_tensor(interaction).to(self.device)

        # 두 서브모델 점수 취득 (각 모델은 자체적으로 user→score 계산을 지원)
        s1 = self.model1.full_sort_predict((user,))  # (B, I)
        s2 = self.model2.full_sort_predict((user,))  # (B, I)

        # 점수 스케일 정규화(옵션)
        s1n = self._normalize(s1)
        s2n = self._normalize(s2)

        # 가중 합산
        scores = self.w1 * s1n + self.w2 * s2n
        return scores

    # ---------- Training API (no-op) ----------
    def calculate_loss(self, interaction, not_train_ui: bool = False):
        """
        추론 전용. 트레이너가 학습 루프를 돌더라도 인터페이스를 맞추기 위해 '0-loss' 반환.
        파라미터는 requires_grad=False 이므로 실제 업데이트는 일어나지 않는다.
        """
        # 장치 일치
        zero = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
        emb = torch.tensor(0.0, device=self.device, requires_grad=True)
        reg = torch.tensor(0.0, device=self.device, requires_grad=True)
        # 트레이너가 (loss, others...)를 언팩할 수도 있으므로 단일 스칼라 반환이 가장 호환성이 높음
        return zero + emb + reg

    def forward(self, *args, **kwargs):
        # MMRec 내부에서 forward를 직접 쓰진 않지만, 호환 목적으로 남겨둠
        return None

    # ---------- Utils ----------
    def _get_user_tensor(self, interaction):
        # dict 또는 tuple 지원
        if isinstance(interaction, dict):
            for k in ("user_id", "user", "uid"):
                if k in interaction:
                    return interaction[k]
            raise KeyError("interaction dict에서 user 텐서를 찾을 수 없습니다. (user_id/user/uid 키 확인)")
        else:
            return interaction[0]

    def _normalize(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (B, I)
        """
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
        """
        다양한 체크포인트 포맷을 견고하게 로드
        """
        state = torch.load(path, map_location=self.device)
        # {'state_dict': ...} 형태 대응
        if isinstance(state, dict) and "state_dict" in state:
            raw = state["state_dict"]
        else:
            raw = state

        # 키 prefix 정리
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