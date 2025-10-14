# models/alignrec_agg.py  (ALIGNREC_ANCHOR_ENSEMBLE_A_LB)
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


class ALIGNREC_ANCHOR_ENSEMBLE_A_LB(GeneralRecommender):
    """
    Linear Blending 앙상블 (+ Grid Search 튜너)
    - 점수 블렌딩: scores = w1 * s1 + (1-w1) * s2
    - 임베딩 블렌딩: u = w1*u1 + (1-w1)*u2, i = w1*i1 + (1-w1)*i2 -> scores = u[user] @ i^T
    - 학습은 없음. 필요 시 검증셋 Recall@K 최대화로 w1을 자동 탐색(tune_weight_by_grid).
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.device = torch.device(config["device"])

        # --- checkpoints & submodels ---
        self.ckpt1 = config['agg_ckpt1'] if 'agg_ckpt1' in config else None
        self.ckpt2 = config['agg_ckpt2'] if 'agg_ckpt2' in config else None
        if not self.ckpt1 or not self.ckpt2:
            raise ValueError("agg_ckpt1, agg_ckpt2 경로를 config에 지정하세요.")

        self.model_name1 = str(config['agg_model1']) if 'agg_model1' in config else "ALIGNREC"
        self.model_name2 = str(config['agg_model2']) if 'agg_model2' in config else "ALIGNREC_ANCHOR"

        # --- weights (초기값; 튜닝 시 덮어씀) ---
        w1 = float(config['agg_weight1']) if 'agg_weight1' in config else 0.5
        w1 = max(0.0, min(1.0, w1))
        self.w1 = w1
        self.w2 = 1.0 - w1

        # --- options ---
        self.blend_target = str(config['agg_blend_target']).lower() if 'agg_blend_target' in config else "score"
        self.norm = str(config['agg_norm']).lower() if 'agg_norm' in config else "zscore"

        self.embed_dim = int(config["embedding_size"]) if "embedding_size" in config else 64
        self.norm_emb = str(config['agg_norm_emb']).lower() if 'agg_norm_emb' in config else "l2"
        self.post_norm_emb = str(config['agg_post_norm_emb']).lower() if 'agg_post_norm_emb' in config else "l2"
        self.cache_embeddings = bool(config["agg_cache_embeddings"]) if "agg_cache_embeddings" in config else True

        # --- submodels ---
        SubModel1 = _MODEL_REGISTRY[self.model_name1]
        SubModel2 = _MODEL_REGISTRY[self.model_name2]
        self.model1 = SubModel1(config, dataset).to(self.device)
        self.model2 = SubModel2(config, dataset).to(self.device)
        self._safe_load(self.model1, self.ckpt1)
        self._safe_load(self.model2, self.ckpt2)
        self.model1.eval(); self.model2.eval()
        for p in self.model1.parameters(): p.requires_grad_(False)
        for p in self.model2.parameters(): p.requires_grad_(False)

        # embedding cache
        self._cached = False
        self._u1 = self._i1 = self._u2 = self._i2 = None

    # -------- trainer hook --------
    def pre_epoch_processing(self):
        self._clear_cache()

    # -------- inference ----------
    @torch.no_grad()
    def full_sort_predict(self, interaction):
        user = self._get_user_tensor(interaction).to(self.device)

        if self.blend_target == "embedding":
            u, i = self._get_blended_embeddings()
            return u[user] @ i.t()

        # score blending
        s1 = self.model1.full_sort_predict((user,))
        s2 = self.model2.full_sort_predict((user,))
        s1n = self._normalize_scores(s1)
        s2n = self._normalize_scores(s2)
        return self.w1 * s1n + self.w2 * s2n

    # -------- no-op training -------
    def calculate_loss(self, interaction, not_train_ui: bool = False):
        z = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
        return z + z

    def forward(self, *args, **kwargs):
        return None

    # -------- embeddings ----------
    @torch.no_grad()
    def _get_base_embeddings(self):
        if self.cache_embeddings and self._cached:
            return self._u1, self._i1, self._u2, self._i2

        u1, i1 = self.model1.forward(self.model1.norm_adj)[:2]
        u2, i2 = self.model2.forward(self.model2.norm_adj)[:2]

        # dim checks
        if u1.shape[1] != self.embed_dim or i1.shape[1] != self.embed_dim:
            raise RuntimeError(f"[DIM MISMATCH] model1 emb_dim={u1.shape[1]} vs {self.embed_dim}")
        if u2.shape[1] != self.embed_dim or i2.shape[1] != self.embed_dim:
            raise RuntimeError(f"[DIM MISMATCH] model2 emb_dim={u2.shape[1]} vs {self.embed_dim}")
        if u1.shape[0] != u2.shape[0] or i1.shape[0] != i2.shape[0]:
            raise RuntimeError(f"[COUNT MISMATCH] u1={u1.shape[0]}, u2={u2.shape[0]}, i1={i1.shape[0]}, i2={i2.shape[0]}")

        if self.norm_emb == "l2":
            u1 = F.normalize(u1, p=2, dim=1); i1 = F.normalize(i1, p=2, dim=1)
            u2 = F.normalize(u2, p=2, dim=1); i2 = F.normalize(i2, p=2, dim=1)

        if self.cache_embeddings:
            self._u1, self._i1, self._u2, self._i2 = u1.detach(), i1.detach(), u2.detach(), i2.detach()
            self._cached = True

        return u1, i1, u2, i2

    @torch.no_grad()
    def _get_blended_embeddings(self):
        u1, i1, u2, i2 = self._get_base_embeddings()
        u = self.w1 * u1 + (1.0 - self.w1) * u2
        i = self.w1 * i1 + (1.0 - self.w1) * i2
        if self.post_norm_emb == "l2":
            u = F.normalize(u, p=2, dim=1)
            i = F.normalize(i, p=2, dim=1)
        return u, i

    def _clear_cache(self):
        self._cached = False
        self._u1 = self._i1 = self._u2 = self._i2 = None

    # -------- Grid Search for w1 --------
    @torch.no_grad()
    def tune_weight_by_grid(self, val_data, k: int = 20, grid=None, use_embedding: bool = False, log_interval: int = 10):
        if grid is None:
            grid = [i / 20.0 for i in range(21)]  # 0.00~1.00 step 0.05

        user_pos = self._build_user_pos_dict(val_data)
        user_list = sorted(user_pos.keys())
        if len(user_list) == 0:
            raise RuntimeError("검증 사용자 집합이 비어있습니다. val split을 확인하세요.")

        best_recall = -1.0
        best_w = self.w1

        if use_embedding or self.blend_target == "embedding":
            u1, i1, u2, i2 = self._get_base_embeddings()

        bs = max(1, getattr(val_data, "batch_size", 256))

        for w in grid:
            w = float(max(0.0, min(1.0, w)))
            recall_sum = 0
            cnt = 0

            if use_embedding or self.blend_target == "embedding":
                u = w * u1 + (1.0 - w) * u2
                i = w * i1 + (1.0 - w) * i2
                if self.post_norm_emb == "l2":
                    u = F.normalize(u, p=2, dim=1); i = F.normalize(i, p=2, dim=1)

                for b_start in range(0, len(user_list), bs):
                    b_users = user_list[b_start:b_start + bs]
                    b_users_t = torch.tensor(b_users, device=self.device, dtype=torch.long)
                    scores = u[b_users_t] @ i.t()
                    topk = torch.topk(scores, k=min(k, scores.shape[1]), dim=1).indices
                    for bi, uid in enumerate(b_users):
                        gt = user_pos.get(uid, set())
                        if not gt:
                            continue
                        pred = set(topk[bi].tolist())
                        hit = len(pred & gt)
                        recall_sum += hit / min(k, len(gt))
                        cnt += 1
            else:
                for b_start in range(0, len(user_list), bs):
                    b_users = user_list[b_start:b_start + bs]
                    b_users_t = torch.tensor(b_users, device=self.device, dtype=torch.long)
                    s1 = self.model1.full_sort_predict((b_users_t,))
                    s2 = self.model2.full_sort_predict((b_users_t,))
                    s1n = self._normalize_scores(s1)
                    s2n = self._normalize_scores(s2)
                    scores = w * s1n + (1.0 - w) * s2n
                    topk = torch.topk(scores, k=min(k, scores.shape[1]), dim=1).indices
                    for bi, uid in enumerate(b_users):
                        gt = user_pos.get(uid, set())
                        if not gt:
                            continue
                        pred = set(topk[bi].tolist())
                        hit = len(pred & gt)
                        recall_sum += hit / min(k, len(gt))
                        cnt += 1

            recall = recall_sum / max(1, cnt)
            if recall > best_recall:
                best_recall = recall
                best_w = w

        self.w1 = float(best_w)
        self.w2 = 1.0 - self.w1
        return {"best_w1": self.w1, "best_recall@{}".format(k): best_recall}

    # -------- GT builders --------
    def _build_user_pos_dict(self, data_loader):
        ds = data_loader.dataset
        # 1) interaction matrix
        try:
            mat = ds.inter_matrix(form='coo').astype('int64')
            user_pos = {}
            for u, i in zip(mat.row.tolist(), mat.col.tolist()):
                user_pos.setdefault(int(u), set()).add(int(i))
            if len(user_pos) > 0:
                return user_pos
        except Exception:
            pass
        # 2) pandas df
        try:
            uid_field = getattr(ds, "uid_field", "user_id")
            iid_field = getattr(ds, "iid_field", "item_id")
            if hasattr(ds, "df"):
                user_pos = {}
                for u, i in zip(ds.df[uid_field].tolist(), ds.df[iid_field].tolist()):
                    user_pos.setdefault(int(u), set()).add(int(i))
                if len(user_pos) > 0:
                    return user_pos
        except Exception:
            pass
        # 3) iterate loader
        user_pos = {}
        try:
            for batch in data_loader:
                if isinstance(batch, dict):
                    u = batch.get('user_id', batch.get('user', batch.get('uid', None)))
                    i = batch.get('item_id', batch.get('item', batch.get('iid', None)))
                    y = batch.get('label', batch.get('y', None))
                else:
                    u, i = batch[0], batch[1]
                    y = batch[2] if len(batch) > 2 else torch.ones_like(u)

                u = u.detach().cpu().long().tolist()
                i = i.detach().cpu().long().tolist()
                if y is not None:
                    y = y.detach().cpu().long().tolist()
                else:
                    y = [1] * len(u)

                for uu, ii, yy in zip(u, i, y):
                    if yy > 0:
                        user_pos.setdefault(int(uu), set()).add(int(ii))
        except Exception:
            pass
        return user_pos

    # -------- utils --------
    def _get_user_tensor(self, interaction):
        if isinstance(interaction, dict):
            for k in ("user_id", "user", "uid"):
                if k in interaction:
                    return interaction[k]
            raise KeyError("interaction dict에서 user 텐서를 찾을 수 없습니다. (user_id/user/uid 키 확인)")
        else:
            return interaction[0]

    def _normalize_scores(self, s: torch.Tensor) -> torch.Tensor:
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