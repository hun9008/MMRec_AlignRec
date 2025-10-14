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


class ALIGNREC_EMB_PROJ_ENSEMBLE_B_1012(GeneralRecommender):
    """
    B (linear projection) 개선판:
      - 두 서브모델 임베딩을 각각 선형투영(Pu*, Pi*) 후 softmax gate로 합산
      - 양 경로 투영 I로 초기화(둘 다 동일 가중), α는 softmax로 w1+w2=1 보장
      - Warm-up 동안 α만 학습, 이후 투영도 함께 학습
      - BPR + 정렬 보조손실(cosine align) + 직교 규제(선택)
      - 정규화 OFF(기본)로 원래 dot-geometry 보존

    Config 키(선택):
      * agg_ckpt1 / agg_ckpt2 / agg_model1 / agg_model2
      * embedding_size
      * agg_cache_embeddings (bool, default True)
      * agg_norm_emb ('none'|'l2', default 'none')      # base 임베딩 취득 직후
      * agg_post_norm ('none'|'l2', default 'none')     # 집계 후
      * agg_score_tau (float, default 1.0)
      * agg_proj_l2 (float, default 1e-5)
      * agg_align_w (float, default 0.05)               # 보조정렬 손실 가중치
      * agg_ortho_w (float, default 1e-4)               # 직교 규제 가중치
      * agg_warmup_epochs (int, default 1)              # warm-up 동안 α만 학습
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.device = torch.device(config["device"])
        self.embed_dim = int(config["embedding_size"]) if "embedding_size" in config else 64

        # checkpoints & submodels
        self.ckpt1 = config['agg_ckpt1'] if 'agg_ckpt1' in config else None
        self.ckpt2 = config['agg_ckpt2'] if 'agg_ckpt2' in config else None
        if not self.ckpt1 or not self.ckpt2:
            raise ValueError("agg_ckpt1, agg_ckpt2 경로를 config에 지정하세요.")

        self.model_name1 = str(config['agg_model1']) if 'agg_model1' in config else "ALIGNREC"
        self.model_name2 = str(config['agg_model2']) if 'agg_model2' in config else "ALIGNREC_ANCHOR"

        # norms (기본은 정규화 OFF)
        self.norm_mode = (str(config['agg_norm_emb']).lower()
                          if 'agg_norm_emb' in config else "none")
        self.post_norm = (str(config['agg_post_norm']).lower()
                          if 'agg_post_norm' in config else "none")

        self.score_tau = float(config['agg_score_tau']) if 'agg_score_tau' in config else 1.0
        self.proj_l2 = float(config['agg_proj_l2']) if 'agg_proj_l2' in config else 1e-5
        self.align_w = float(config['agg_align_w']) if 'agg_align_w' in config else 0.05
        self.ortho_w = float(config['agg_ortho_w']) if 'agg_ortho_w' in config else 1e-4
        self.warmup_epochs = int(config['agg_warmup_epochs']) if 'agg_warmup_epochs' in config else 1

        # submodels
        SubModel1 = _MODEL_REGISTRY[self.model_name1]
        SubModel2 = _MODEL_REGISTRY[self.model_name2]
        self.model1 = SubModel1(config, dataset).to(self.device)
        self.model2 = SubModel2(config, dataset).to(self.device)
        self._safe_load(self.model1, self.ckpt1)
        self._safe_load(self.model2, self.ckpt2)
        self.model1.eval(); self.model2.eval()
        for p in self.model1.parameters(): p.requires_grad_(False)
        for p in self.model2.parameters(): p.requires_grad_(False)

        # projections (learnable)
        self.Pu1 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.Pu2 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.Pi1 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.Pi2 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # gate: softmax(β) → α 보장 (w1+w2=1, 음수 없음)
        self.beta_u = nn.Parameter(torch.zeros(2))  # [0.0, 0.0] -> α=[0.5,0.5]
        self.beta_i = nn.Parameter(torch.zeros(2))

        # identity init (양 경로 동일 가중으로 시작)
        self._init_proj_identity()

        # embedding cache
        self.cache_embeddings = bool(config["agg_cache_embeddings"]) if "agg_cache_embeddings" in config else True
        self._cached = False
        self._u1 = self._i1 = self._u2 = self._i2 = None

        # epoch 카운터 (warmup 제어)
        self._epoch = -1  # pre_epoch_processing()에서 0부터 시작

    # --------- Trainer hook ---------
    def pre_epoch_processing(self):
        # 에폭 증가 & 캐시 초기화
        self._epoch += 1
        self._clear_cache()

        # warm-up: α만 학습, 투영은 고정
        proj_trainable = (self._epoch >= self.warmup_epochs)
        for m in [self.Pu1, self.Pu2, self.Pi1, self.Pi2]:
            for p in m.parameters():
                p.requires_grad_(proj_trainable)

    # ---------- Inference ----------
    @torch.no_grad()
    def full_sort_predict(self, interaction):
        user = self._get_user_tensor(interaction).to(self.device)
        u_agg, i_agg = self._aggregate()

        scores = u_agg[user] @ i_agg.t()
        if self.score_tau != 1.0:
            scores = scores / self.score_tau
        return scores

    # ---------- Training ----------
    def calculate_loss(self, interaction, not_train_ui: bool = False):
        users = interaction[0].to(self.device)
        pos_items = interaction[1].to(self.device)
        neg_items = interaction[2].to(self.device)

        # 집계 임베딩
        (u1, i1, u2, i2) = self._get_base_embeddings()
        alpha_u = torch.softmax(self.beta_u, dim=0)  # [2]
        alpha_i = torch.softmax(self.beta_i, dim=0)  # [2]

        u_agg = alpha_u[0]*self.Pu1(u1) + alpha_u[1]*self.Pu2(u2)
        i_agg = alpha_i[0]*self.Pi1(i1) + alpha_i[1]*self.Pi2(i2)

        u = u_agg[users]
        pos = i_agg[pos_items]
        neg = i_agg[neg_items]

        # BPR
        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

        loss = bpr

        # 정렬 보조손실 (cosine align)
        if self.align_w > 0:
            with torch.no_grad():
                # 샘플링(선택): 너무 크면 학습 느려질 수 있어 필요시 subsample
                pass
            align_u = 1 - F.cosine_similarity(self.Pu1(u1[users]), self.Pu2(u2[users]), dim=1).mean()
            align_i = 1 - F.cosine_similarity(self.Pi1(i1[pos_items]), self.Pi2(i2[pos_items]), dim=1).mean()
            loss = loss + self.align_w * (align_u + align_i)

        # 직교 규제 (선택)
        if self.ortho_w > 0:
            def ortho(m):
                M = m.weight
                I = torch.eye(M.size(0), device=M.device)
                return (M.t() @ M - I).pow(2).sum()
            loss = loss + self.ortho_w * (
                ortho(self.Pu1) + ortho(self.Pu2) + ortho(self.Pi1) + ortho(self.Pi2)
            )

        # L2 규제(약하게)
        if self.proj_l2 > 0:
            reg = 0.0
            for m in [self.Pu1, self.Pu2, self.Pi1, self.Pi2]:
                reg = reg + m.weight.pow(2).sum()
            loss = loss + self.proj_l2 * reg

        return loss

    def forward(self, *args, **kwargs):
        return None

    # ---------- Internals ----------
    @torch.no_grad()
    def _get_base_embeddings(self):
        if self.cache_embeddings and self._cached:
            return self._u1, self._i1, self._u2, self._i2

        u1, i1 = self.model1.forward(self.model1.norm_adj)[:2]
        u2, i2 = self.model2.forward(self.model2.norm_adj)[:2]

        # dim checks
        if u1.shape[1] != self.embed_dim or i1.shape[1] != self.embed_dim:
            raise RuntimeError(f"[DIM MISMATCH] model1 emb_dim={u1.shape[1]} vs expected {self.embed_dim}")
        if u2.shape[1] != self.embed_dim or i2.shape[1] != self.embed_dim:
            raise RuntimeError(f"[DIM MISMATCH] model2 emb_dim={u2.shape[1]} vs expected {self.embed_dim}")
        if u1.shape[0] != u2.shape[0] or i1.shape[0] != i2.shape[0]:
            raise RuntimeError(f"[COUNT MISMATCH] users/items mismatch: u1={u1.shape[0]}, u2={u2.shape[0]}, i1={i1.shape[0]}, i2={i2.shape[0]}")

        # base norm (기본 none)
        if self.norm_mode == "l2":
            u1 = F.normalize(u1, p=2, dim=1); i1 = F.normalize(i1, p=2, dim=1)
            u2 = F.normalize(u2, p=2, dim=1); i2 = F.normalize(i2, p=2, dim=1)

        if self.cache_embeddings:
            self._u1, self._i1, self._u2, self._i2 = u1.detach(), i1.detach(), u2.detach(), i2.detach()
            self._cached = True

        return u1, i1, u2, i2

    def _aggregate(self):
        u1, i1, u2, i2 = self._get_base_embeddings()
        alpha_u = torch.softmax(self.beta_u, dim=0)  # [2]
        alpha_i = torch.softmax(self.beta_i, dim=0)  # [2]
        u_agg = alpha_u[0]*self.Pu1(u1) + alpha_u[1]*self.Pu2(u2)
        i_agg = alpha_i[0]*self.Pi1(i1) + alpha_i[1]*self.Pi2(i2)

        if self.post_norm == "l2":
            u_agg = F.normalize(u_agg, p=2, dim=1)
            i_agg = F.normalize(i_agg, p=2, dim=1)
        return u_agg, i_agg

    def _clear_cache(self):
        self._cached = False
        self._u1 = self._i1 = self._u2 = self._i2 = None

    def _init_proj_identity(self):
        with torch.no_grad():
            eye = torch.eye(self.embed_dim, device=self.device)
            for m in [self.Pu1, self.Pu2, self.Pi1, self.Pi2]:
                m.weight.copy_(eye)

    def _get_user_tensor(self, interaction):
        if isinstance(interaction, dict):
            for k in ("user_id", "user", "uid"):
                if k in interaction:
                    return interaction[k]
            raise KeyError("interaction dict에서 user 텐서를 찾을 수 없습니다. (user_id/user/uid 키 확인)")
        else:
            return interaction[0]

    def _safe_load(self, model: nn.Module, path: str):
        state = torch.load(path, map_location=self.device)
        raw = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        if not isinstance(raw, dict):
            raise RuntimeError("Unexpected checkpoint format (expected dict)")

        src = {}
        for k, v in raw.items():
            kk = k
            if kk.startswith("model."):  kk = kk[6:]
            if kk.startswith("module."): kk = kk[7:]
            src[kk] = v

        dst = model.state_dict()
        new_state, mismatched = {}, []
        for k, v in src.items():
            if k in dst:
                if tuple(dst[k].shape) == tuple(v.shape):
                    new_state[k] = v
                else:
                    mismatched.append((k, tuple(v.shape), tuple(dst[k].shape)))

        if mismatched:
            msg = "\n".join([f"{k}: ckpt{cs} vs model{ms}" for k, cs, ms in mismatched])
            raise RuntimeError(f"[CKPT SHAPE MISMATCH]\n{msg}\n-> ckpt/데이터셋/모델 버전을 확인하세요.")

        model.load_state_dict(new_state, strict=False)