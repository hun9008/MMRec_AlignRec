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
    from models.alignrec import ALIGNREC
    from models.alignrec_anchor import ALIGNREC_ANCHOR


_MODEL_REGISTRY = {
    "ALIGNREC": ALIGNREC,
    "ALIGNREC_ANCHOR": ALIGNREC_ANCHOR,
}


class ALIGNREC_EMB_GATE_ENSEMBLE_C_1012(GeneralRecommender):
    """
    옵션 C: 게이팅/주목 가중 합(learned gating)
    - 두 사전학습 모델의 user/item 임베딩 (u1,i1), (u2,i2)를 받아
      유저/아이템별 게이트 g_u, g_i를 학습해 가중합으로 집계.
    - SubModel 파라미터는 동결, 게이트(작은 MLP)만 학습.
    - calculate_loss: BPR(pairwise)로 게이트를 학습.
    - full_sort_predict: 게이트 집계 임베딩으로 점수 계산.

    옵션:
      * agg_norm_emb:     base 임베딩 정규화   (none|l2) [default: l2]
      * agg_post_norm:    집계 후 정규화       (none|l2) [default: l2]
      * agg_score_tau:    점수 온도 스케일     (float)   [default: 1.0]
      * agg_cache_embeddings: 임베딩 캐시 사용 (bool)    [default: True]
      * agg_gate_hidden:  게이트 내부 차원     (int)     [default: 0 → 선형만]
      * agg_gate_bias:    게이트 바이어스 초기값 (float) [default: 0.0]
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ---- 필수 설정 ----
        self.device = torch.device(config["device"])
        self.embed_dim = int(config["embedding_size"]) if "embedding_size" in config else 64

        # 체크포인트 / 서브모델 선택
        self.ckpt1 = config["agg_ckpt1"] if "agg_ckpt1" in config else None
        self.ckpt2 = config["agg_ckpt2"] if "agg_ckpt2" in config else None
        if not self.ckpt1 or not self.ckpt2:
            raise ValueError("agg_ckpt1, agg_ckpt2 경로를 config에 지정하세요.")

        self.model_name1 = str(config["agg_model1"]) if "agg_model1" in config else "ALIGNREC"
        self.model_name2 = str(config["agg_model2"]) if "agg_model2" in config else "ALIGNREC_ANCHOR"

        # 정규화/스케일
        self.norm_mode = str(config["agg_norm_emb"]).lower() if "agg_norm_emb" in config else "l2"
        self.post_norm = str(config["agg_post_norm"]).lower() if "agg_post_norm" in config else "l2"
        self.score_tau = float(config["agg_score_tau"]) if "agg_score_tau" in config else 1.0

        # 게이트 구조 옵션
        self.gate_hidden = int(config["agg_gate_hidden"]) if "agg_gate_hidden" in config else 0
        gate_bias_init = float(config["agg_gate_bias"]) if "agg_gate_bias" in config else 0.0

        # ---- Sub-models 로드 & 동결 ----
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

        # ---- Gating heads (learnable) ----
        # 입력: [e1; e2] ∈ R^{2d} → sigmoid ∈ (0,1)
        in_dim = 2 * self.embed_dim
        if self.gate_hidden and self.gate_hidden > 0:
            self.gate_u = nn.Sequential(
                nn.Linear(in_dim, self.gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.gate_hidden, 1),
                nn.Sigmoid(),
            )
            self.gate_i = nn.Sequential(
                nn.Linear(in_dim, self.gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.gate_hidden, 1),
                nn.Sigmoid(),
            )
        else:
            self.gate_u = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())
            self.gate_i = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

        # 게이트 바이어스 초기화(선호도 사전 설정)
        with torch.no_grad():
            # 마지막 Linear의 bias만 찾아서 설정
            for mod in [self.gate_u, self.gate_i]:
                # Sequential 마지막에서 Linear를 찾음
                last_linear = None
                for m in mod:
                    if isinstance(m, nn.Linear):
                        last_linear = m
                if last_linear is not None and last_linear.bias is not None:
                    # bias>0 → g>0.5 (model1 비중↑), bias<0 → g<0.5
                    last_linear.bias.fill_(gate_bias_init)

        # 임베딩 캐시
        self.cache_embeddings = bool(config["agg_cache_embeddings"]) if "agg_cache_embeddings" in config else True
        self._cached = False
        self._u1 = None
        self._i1 = None
        self._u2 = None
        self._i2 = None

    # --------- Trainer 훅: 에폭 시작 전에 캐시 초기화 ---------
    def pre_epoch_processing(self):
        self._clear_cache()

    # ---------- Embedding 취득 ----------
    @torch.no_grad()
    def _get_base_embeddings(self):
        """
        두 서브모델의 (user_emb, item_emb) 취득.
        각 서브모델은 forward(norm_adj)[:2] 형태로 (users, items)를 반환한다고 가정.
        """
        if self.cache_embeddings and self._cached:
            return self._u1, self._i1, self._u2, self._i2

        # model1
        out1 = self.model1.forward(self.model1.norm_adj)
        u1, i1 = out1[0], out1[1]

        # model2
        out2 = self.model2.forward(self.model2.norm_adj)
        u2, i2 = out2[0], out2[1]

        # 안전 체크
        if u1.shape[1] != self.embed_dim or i1.shape[1] != self.embed_dim:
            raise RuntimeError(f"[DIM MISMATCH] model1 emb_dim={u1.shape[1]} vs expected {self.embed_dim}")
        if u2.shape[1] != self.embed_dim or i2.shape[1] != self.embed_dim:
            raise RuntimeError(f"[DIM MISMATCH] model2 emb_dim={u2.shape[1]} vs expected {self.embed_dim}")
        if u1.shape[0] != u2.shape[0] or i1.shape[0] != i2.shape[0]:
            raise RuntimeError(f"[COUNT MISMATCH] users/items count differ: "
                               f"u1={u1.shape[0]}, u2={u2.shape[0]}, i1={i1.shape[0]}, i2={i2.shape[0]}")

        # base 임베딩 정규화
        if self.norm_mode == "l2":
            u1 = F.normalize(u1, p=2, dim=1)
            i1 = F.normalize(i1, p=2, dim=1)
            u2 = F.normalize(u2, p=2, dim=1)
            i2 = F.normalize(i2, p=2, dim=1)

        if self.cache_embeddings:
            self._u1, self._i1, self._u2, self._i2 = u1.detach(), i1.detach(), u2.detach(), i2.detach()
            self._cached = True

        return u1, i1, u2, i2

    def _clear_cache(self):
        self._cached = False
        self._u1 = self._i1 = self._u2 = self._i2 = None

    # ---------- Aggregation (Gating) ----------
    def _aggregate(self):
        """
        게이트로 가중 합:
          g_u = sigmoid( W_u [u1;u2] )
          u   = g_u * u1 + (1 - g_u) * u2
          g_i = sigmoid( W_i [i1;i2] )
          i   = g_i * i1 + (1 - g_i) * i2
        이후 post-norm(none|l2) 적용.
        """
        u1, i1, u2, i2 = self._get_base_embeddings()

        gu = self.gate_u(torch.cat([u1, u2], dim=1))  # (n_users, 1)
        gi = self.gate_i(torch.cat([i1, i2], dim=1))  # (n_items, 1)

        u_agg = gu * u1 + (1.0 - gu) * u2
        i_agg = gi * i1 + (1.0 - gi) * i2

        if self.post_norm == "l2":
            u_agg = F.normalize(u_agg, p=2, dim=1)
            i_agg = F.normalize(i_agg, p=2, dim=1)

        return u_agg, i_agg

    # ---------- Inference ----------
    @torch.no_grad()
    def full_sort_predict(self, interaction):
        """
        interaction: (user_tensor, ) 또는 {'user_id': tensor}
        return: (B, n_items)
        """
        user = self._get_user_tensor(interaction).to(self.device)
        u_agg, i_agg = self._aggregate()
        scores = u_agg[user] @ i_agg.t()
        if self.score_tau != 1.0:
            scores = scores / self.score_tau
        return scores

    # ---------- Training (Gates-Only) ----------
    def calculate_loss(self, interaction, not_train_ui: bool = False):
        """
        게이트만 BPR로 미세학습. 서브모델 파라미터는 requires_grad=False.
        interaction: (users, pos_items, neg_items)
        """
        users = interaction[0].to(self.device)
        pos_items = interaction[1].to(self.device)
        neg_items = interaction[2].to(self.device)

        u_agg, i_agg = self._aggregate()

        u = u_agg[users]
        pos_i = i_agg[pos_items]
        neg_i = i_agg[neg_items]

        pos_scores = (u * pos_i).sum(dim=1)
        neg_scores = (u * neg_i).sum(dim=1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

        # 게이트 가중치에 대한 L2 (게이트가 과도하게 커지지 않도록)
        # 게이트가 Linear만 있는 경우 마지막 Linear의 weight만 규제
        gate_reg = 0.0
        for mod in [self.gate_u, self.gate_i]:
            for m in mod:
                if isinstance(m, nn.Linear):
                    gate_reg = gate_reg + m.weight.pow(2).sum()
        gate_reg = 1e-4 * gate_reg  # 필요 시 config로 노출 가능

        loss = bpr + gate_reg
        return loss

    def forward(self, *args, **kwargs):
        # 트레이너 호환용
        return None

    # ---------- Utils ----------
    def _get_user_tensor(self, interaction):
        if isinstance(interaction, dict):
            for k in ("user_id", "user", "uid"):
                if k in interaction:
                    return interaction[k]
            raise KeyError("interaction dict에서 user 텐서를 찾을 수 없습니다. (user_id/user/uid 키 확인)")
        else:
            return interaction[0]

    def _safe_load(self, model: nn.Module, path: str):
        """
        체크포인트 shape 엄격 검증. 불일치 시 즉시 에러.
        """
        state = torch.load(path, map_location=self.device)
        raw = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state

        if not isinstance(raw, dict):
            raise RuntimeError("Unexpected checkpoint format (expected dict)")

        # prefix 정리
        src = {}
        for k, v in raw.items():
            kk = k
            if kk.startswith("model."):
                kk = kk[6:]
            if kk.startswith("module."):
                kk = kk[7:]
            src[kk] = v

        dst = model.state_dict()
        new_state = {}
        mismatched = []

        for k, v in src.items():
            if k in dst:
                if dst[k].shape == v.shape:
                    new_state[k] = v
                else:
                    mismatched.append((k, tuple(v.shape), tuple(dst[k].shape)))
            # dst에 없는 키는 무시

        if mismatched:
            msg = "\n".join([f"{k}: ckpt{cs} vs model{ms}" for k, cs, ms in mismatched])
            raise RuntimeError(f"[CKPT SHAPE MISMATCH]\n{msg}\n-> ckpt/데이터셋/모델 버전을 확인하세요.")

        model.load_state_dict(new_state, strict=False)