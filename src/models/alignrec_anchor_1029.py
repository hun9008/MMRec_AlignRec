import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_max
except Exception:
    pass

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss
from utils.utils import build_sim, build_knn_normalized_graph


class ALIGNREC_ANCHOR_1029(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.sparse = True
        self.cl_loss = config['cl_loss']  # alpha
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.lambda_weight = config['lambda_weight']  # lambda
        self.desc = config['desc']
        self.use_ln = config['use_ln']
        self.sim_weight = config['sim_weight']  # beta
        self.ui_cosine_loss_weight = config['ui_cosine_loss_weight']
        self.recover_lambda = config['recover_lambda']

        # 새로 추가된 계층형 alignment loss 가중치
        self.gamma_mm = config['gamma_mm'] if 'gamma_mm' in config else 1.0
        self.gamma_s = config['gamma_s'] if 'gamma_s' in config else 1.0

        self.use_cross_att = False
        self.use_user_history = False
        self.add_user_history_after_content_embs = False
        self.reg_loss = EmbLoss()

        # 최종 임베딩에 복원된 MM만 더할지 제어
        self.use_recovered_mm_for_final = True

        # interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # ====== 기본 user/item ID 임베딩 ======
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # ====== Projection MLPs (alignment 공간) ======
        self.W_id_i = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        self.W_mm_i = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        self.W_v = nn.Sequential(
            nn.Linear(self.v_feat.shape[1], self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        self.W_t = nn.Sequential(
            nn.Linear(self.t_feat.shape[1], self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        # projection-space(mm) → ID-space 복원 디코더
        self.D_mm2id = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        # init linear weights
        for block in [self.W_id_i, self.W_mm_i, self.W_v, self.W_t, self.D_mm2id]:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

        # LightGCN graph 준비
        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(
            self.device
        )

        # ====== 모달별 item-item graph 준비 ======
        if self.mm_feat is not None:
            self.mm_embedding = nn.Embedding.from_pretrained(
                self.mm_feat, freeze=False
            )
            mm_adj = build_sim(self.mm_embedding.weight.detach())
            mm_adj = build_knn_normalized_graph(
                mm_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type="sym"
            )
            self.mm_original_adj = mm_adj.to(self.device)

            if self.use_ln:
                self.mm_ln = nn.LayerNorm(self.mm_feat.shape[1])
            self.mm_trs = nn.Linear(self.mm_feat.shape[1], self.embedding_dim)

        if self.t_feat is not None:
            self.t_embedding = nn.Embedding.from_pretrained(
                self.t_feat, freeze=False
            )
            t_adj = build_sim(self.t_embedding.weight.detach())
            t_adj = build_knn_normalized_graph(
                t_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type="sym"
            )
            self.t_original_adj = t_adj.to(self.device)

            if self.use_ln:
                self.t_ln = nn.LayerNorm(self.t_feat.shape[1])
            self.t_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        if self.v_feat is not None:
            self.v_embedding = nn.Embedding.from_pretrained(
                self.v_feat, freeze=False
            )
            v_adj = build_sim(self.v_embedding.weight.detach())
            v_adj = build_knn_normalized_graph(
                v_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type="sym"
            )
            self.v_original_adj = v_adj.to(self.device)

            if self.use_ln:
                self.v_ln = nn.LayerNorm(self.v_feat.shape[1])
            self.v_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        # (옵션) attention 관련
        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False),
        )

        self.gate_mm_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid(),
        )
        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid(),
        )

        self.use_bce = False
        print("use_bce", self.use_bce)
        if self.use_bce:
            self.sigbce_loss = nn.BCEWithLogitsLoss()

        self.side_emb_div = config["side_emb_div"]  # default 0

        # ====== user history decoder (옵션) ======
        self.use_hist_decoder = config["use_hist_decoder"]
        if self.use_hist_decoder:
            self.user_topk_hist = self.init_user_hist_info(dataset)
            self.query = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.key = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.value = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.hist_ln1 = nn.LayerNorm(self.embedding_dim)

        self.test_arch1 = config["test_arch1"]
        if self.test_arch1:
            self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
            nn.init.xavier_normal_(self.predictor.weight)

        self.ui_cosine_loss = config["ui_cosine_loss"]

        self.tau = 0.5  # graded contrastive temperature

    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self):
        # LightGCN bipartite adjacency
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32,
        )
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[: self.n_users, self.n_users :] = R
        adj_mat[self.n_users :, : self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()

        # user->item block for later propagation
        self.R = norm_adj_mat[: self.n_users, self.n_users :]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, users=None, train=False):
        """
        train=True일 때:
            return (...,
                    proj_item_dict, raw_item_dict)
        train=False일 때:
            return (... ) without dicts
        """

        # ====== MM feats to (n_items, d) ======
        if self.mm_feat is not None:
            if self.use_ln:
                mm_feats0 = self.mm_ln(self.mm_embedding.weight)
            else:
                mm_feats0 = self.mm_embedding.weight
            mm_feats = self.mm_trs(mm_feats0)  # (n_items, d)
        else:
            mm_feats = None

        if mm_feats is not None:
            mm_item_embeds = torch.multiply(
                self.item_id_embedding.weight, self.gate_v(mm_feats)
            ).to(self.device)
        else:
            # fallback: just ID emb if mm_feats is None
            mm_item_embeds = self.item_id_embedding.weight.to(self.device)

        # ====== Text feats ======
        if self.t_feat is not None:
            if self.use_ln:
                t_feats0 = self.t_ln(self.t_embedding.weight)
            else:
                t_feats0 = self.t_embedding.weight
            t_feats = self.t_trs(t_feats0)  # (n_items, d)
        else:
            t_feats = None

        if t_feats is not None:
            t_item_embeds = torch.multiply(
                self.item_id_embedding.weight, self.gate_v(t_feats)
            ).to(self.device)
        else:
            t_item_embeds = self.item_id_embedding.weight.to(self.device)

        # ====== Vision feats ======
        if self.v_feat is not None:
            if self.use_ln:
                v_feats0 = self.v_ln(self.v_embedding.weight)
            else:
                v_feats0 = self.v_embedding.weight
            v_feats = self.v_trs(v_feats0)  # (n_items, d)
        else:
            v_feats = None

        if v_feats is not None:
            v_item_embeds = torch.multiply(
                self.item_id_embedding.weight, self.gate_v(v_feats)
            ).to(self.device)
        else:
            v_item_embeds = self.item_id_embedding.weight.to(self.device)

        # ====== UI view (LightGCN propagation over user-item) ======
        item_embeds = self.item_id_embedding.weight.to(self.device)  # (n_items,d)
        user_embeds = self.user_embedding.weight.to(self.device)  # (n_users,d)

        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)  # (U+I,d)
        all_embeddings = [ego_embeddings]

        for _ in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1, keepdim=False)
        content_embeds = all_embeddings  # (U+I, d)

        # ====== item-item graph propagation for mm/t/v ======
        # mm branch
        mm_item_prop = mm_item_embeds
        for _ in range(self.n_layers):
            mm_item_prop = torch.sparse.mm(self.mm_original_adj, mm_item_prop)
        mm_user_embeds = torch.sparse.mm(self.R.to(self.device), mm_item_prop)
        mm_embeds = torch.cat([mm_user_embeds, mm_item_prop], dim=0)  # (U+I,d)

        # text branch
        t_item_prop = t_item_embeds
        for _ in range(self.n_layers):
            t_item_prop = torch.sparse.mm(self.t_original_adj, t_item_prop)
        t_user_embeds = torch.sparse.mm(self.R.to(self.device), t_item_prop)
        t_embeds = torch.cat([t_user_embeds, t_item_prop], dim=0)  # (U+I,d)

        # vision branch
        v_item_prop = v_item_embeds
        for _ in range(self.n_layers):
            v_item_prop = torch.sparse.mm(self.v_original_adj, v_item_prop)
        v_user_embeds = torch.sparse.mm(self.R.to(self.device), v_item_prop)
        v_embeds = torch.cat([v_user_embeds, v_item_prop], dim=0)  # (U+I,d)

        # ====== Split user/item ======
        content_embeds_user, content_embeds_items = torch.split(
            content_embeds, [self.n_users, self.n_items], dim=0
        )
        mm_embeds_user, mm_embeds_items = torch.split(
            mm_embeds, [self.n_users, self.n_items], dim=0
        )
        t_embeds_user, t_embeds_items = torch.split(
            t_embeds, [self.n_users, self.n_items], dim=0
        )
        v_embeds_user, v_embeds_items = torch.split(
            v_embeds, [self.n_users, self.n_items], dim=0
        )

        # ====== Projection space ======
        h_id_i_fusion = self.W_id_i(content_embeds_items)  # (I,d)
        h_mm_i_fusion = self.W_mm_i(mm_embeds_items)  # (I,d)
        h_t_i_fusion = self.W_t(self.t_feat)  # (I,d) uses raw t_feat
        h_v_i_fusion = self.W_v(self.v_feat)  # (I,d) uses raw v_feat

        # 복원
        mm_rec_i = self.D_mm2id(h_mm_i_fusion)  # (I,d)

        # ====== 최종 user/item embeddings for recommendation ======
        if self.use_recovered_mm_for_final:
            all_embeds_sum = content_embeds + mm_embeds + t_embeds  # (U+I,d)
            final_user_e, final_item_e = torch.split(
                all_embeds_sum, [self.n_users, self.n_items], dim=0
            )
            final_item_e = final_item_e + self.recover_lambda * mm_rec_i
        else:
            all_embeds_sum = content_embeds + mm_embeds + t_embeds + v_embeds
            final_user_e, final_item_e = torch.split(
                all_embeds_sum, [self.n_users, self.n_items], dim=0
            )

        # ====== optional user history decoder ======
        if self.use_hist_decoder and train:
            hist_seq = mm_item_prop[self.user_topk_hist[users], :]
            hist_seq = torch.where(
                (self.user_topk_hist[users].unsqueeze(-1) == -1),
                torch.zeros_like(hist_seq),
                hist_seq,
            )
            score = torch.bmm(
                self.query(hist_seq),
                self.key(hist_seq).transpose(1, 2),
            ) / np.sqrt(self.embedding_dim)
            score.masked_fill_(
                (self.user_topk_hist[users] == -1).view(-1, 1, 10), -float("Inf")
            )
            attn = F.softmax(score, -1)
            context = torch.bmm(attn, self.value(hist_seq))
            hist_hid = self.hist_ln1(hist_seq + context)  # (B, topk, d)

        if train:
            proj_item_dict = {
                "mm_proj": h_mm_i_fusion,  # (I,d)
                "t_proj": h_t_i_fusion,
                "v_proj": h_v_i_fusion,
                "id_proj": h_id_i_fusion,
            }
            raw_item_dict = {
                "mm_raw": mm_embeds_items,  # (I,d) graph-propagated mm
                "t_raw": t_embeds_items,
                "v_raw": v_embeds_items,
            }

            if self.use_hist_decoder:
                return (
                    final_user_e,
                    final_item_e,
                    mm_embeds,
                    content_embeds,
                    t_embeds,
                    v_embeds,
                    hist_hid[:, -1, :],
                    proj_item_dict,
                    raw_item_dict,
                )
            else:
                return (
                    final_user_e,
                    final_item_e,
                    mm_embeds,
                    content_embeds,
                    t_embeds,
                    v_embeds,
                    proj_item_dict,
                    raw_item_dict,
                )

        # eval path
        return final_user_e, final_item_e, mm_embeds, content_embeds, t_embeds, v_embeds

    # ======================
    # Loss utilities
    # ======================

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        regularizer = (
            0.5 * (users ** 2).sum()
            + 0.5 * (pos_items ** 2).sum()
            + 0.5 * (neg_items ** 2).sum()
        )
        regularizer = regularizer / self.batch_size
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        emb_loss = self.lambda_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)  # (B,)
        pos_score = torch.exp(pos_score / temperature)  # (B,)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))  # (B,B)
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)  # (B,)
        cl_loss = -torch.log(pos_score / (ttl_score + 1e-8) + 1e-8)
        return torch.mean(cl_loss)

    def sim_loss(self, embedding, sim):
        embedding_sim = torch.mm(embedding, embedding.t())
        sim_loss_val = self.reg_loss(embedding_sim - sim.detach())
        return sim_loss_val

    def sim_sigmoid_loss(self, embedding, sim):
        embedding_sim = torch.mm(embedding, embedding.t())
        logit_emb_sim = torch.reshape(embedding_sim, (-1, 1))
        logit_sim = torch.reshape(sim, (-1, 1))
        target = torch.sigmoid(logit_sim)
        sim_loss_val = self.sigbce_loss(logit_emb_sim, target)
        return sim_loss_val

    # ======================
    # Graded similarity helpers (AI-MM / AI-S)
    # ======================

    def _cosine_matrix(self, x):
        # x: (B_i, d)
        x_norm = F.normalize(x, dim=1)
        return torch.matmul(x_norm, x_norm.T)  # (B_i,B_i)

    def _topk_neighbors(self, sim_mat, k):
        # sim_mat: (B_i,B_i)
        mask_self = torch.eye(sim_mat.size(0), device=sim_mat.device).bool()
        sim_masked = sim_mat.masked_fill(mask_self, -1e9)
        _, idx = torch.topk(sim_masked, k, dim=1)
        return idx  # (B_i,k)

    def _build_graded_sets(self, mm_sim, t_sim, K_modal):
        """
        mm_sim, t_sim: (B_i,B_i)
        return:
          R_sets[i]: set of indices similar in BOTH mm and t (multi-modal strong)
          T_sets[i]: set of indices similar in exactly one of them (single-modal)
          N_sets[i]: rest except self
        """
        B_i = mm_sim.size(0)
        mm_topk = self._topk_neighbors(mm_sim, K_modal)  # (B_i,K)
        t_topk = self._topk_neighbors(t_sim, K_modal)  # (B_i,K)

        all_idx = set(range(B_i))
        R_sets, T_sets, N_sets = [], [], []

        for a in range(B_i):
            mm_set = set(mm_topk[a].tolist())
            t_set = set(t_topk[a].tolist())

            R = mm_set & t_set  # strong multimodal
            T = mm_set ^ t_set  # single-modal only
            N = all_idx - R - T - {a}

            R_sets.append(R)
            T_sets.append(T)
            N_sets.append(N)

        return R_sets, T_sets, N_sets

    def _phi(self, anchor_vec, others, tau):
        # anchor_vec: (d,)
        # others: (n,d)
        if others.size(0) == 0:
            return torch.tensor(0.0, device=anchor_vec.device)
        a_norm = F.normalize(anchor_vec.unsqueeze(0), dim=1)  # (1,d)
        b_norm = F.normalize(others, dim=1)  # (n,d)
        cos_sim = torch.matmul(a_norm, b_norm.T).squeeze(0)  # (n,)
        return torch.exp(cos_sim / tau).sum()  # scalar

    def _graded_contrastive_loss(self, emb_items, R_sets, T_sets, N_sets, tau):
        """
        emb_items: (B_i,d)  -> projection 후 임베딩 (하나의 modality view)
        R_sets/T_sets/N_sets: list[set(int)]
        Return:
            ai_mm: DA-MRS style AI-MM
            ai_s:  DA-MRS style AI-S
        """
        B_i = emb_items.size(0)
        losses_ai_mm = []
        losses_ai_s = []

        for a in range(B_i):
            R_idx = list(R_sets[a])
            T_idx = list(T_sets[a])
            N_idx = list(N_sets[a])

            # AI-S 최소 조건: T and N 존재
            # AI-MM 최소 조건: R 존재
            if len(T_idx) == 0 or len(N_idx) == 0:
                # skip if no contrastive structure
                continue

            anchor = emb_items[a]  # (d,)
            R_phi = (
                self._phi(anchor, emb_items[R_idx], tau)
                if len(R_idx) > 0
                else torch.tensor(0.0, device=emb_items.device)
            )
            T_phi = self._phi(anchor, emb_items[T_idx], tau)
            N_phi = self._phi(anchor, emb_items[N_idx], tau)

            # AI-MM: multimodal-strong vs others
            if len(R_idx) > 0:
                numer_mm = R_phi + 1e-8
                denom_mm = R_phi + T_phi + N_phi + 1e-8
                loss_ai_mm_a = -torch.log(numer_mm / denom_mm)
                losses_ai_mm.append(loss_ai_mm_a)

            # AI-S: single-modal vs dissimilar
            numer_s = T_phi + 1e-8
            denom_s = T_phi + N_phi + 1e-8
            loss_ai_s_a = -torch.log(numer_s / denom_s)
            losses_ai_s.append(loss_ai_s_a)

        if len(losses_ai_mm) == 0:
            ai_mm = torch.tensor(0.0, device=emb_items.device)
        else:
            ai_mm = torch.stack(losses_ai_mm).mean()

        if len(losses_ai_s) == 0:
            ai_s = torch.tensor(0.0, device=emb_items.device)
        else:
            ai_s = torch.stack(losses_ai_s).mean()

        return ai_mm, ai_s

    # ======================
    # Main training loss
    # ======================

    def calculate_loss(self, interaction, not_train_ui=False):
        users = interaction[0]  # (B,)
        pos_items = interaction[1]  # (B,)
        neg_items = interaction[2]  # (B,)

        if self.use_hist_decoder:
            (
                ua_embeddings,
                ia_embeddings,
                side_embeds,
                content_embeds,
                t_embeds,
                v_embeds,
                user_hist_seq,
                proj_item_dict,
                raw_item_dict,
            ) = self.forward(self.norm_adj, users, train=True)
        else:
            (
                ua_embeddings,
                ia_embeddings,
                side_embeds,
                content_embeds,
                t_embeds,
                v_embeds,
                proj_item_dict,
                raw_item_dict,
            ) = self.forward(self.norm_adj, users, train=True)

        # --- BPR core ---
        u_g_embeddings = ua_embeddings[users]  # (B,d)
        pos_i_g_embeddings = ia_embeddings[pos_items]  # (B,d)
        neg_i_g_embeddings = ia_embeddings[neg_items]  # (B,d)

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )

        # --- 분해된 뷰 ---
        side_embeds_users, side_embeds_items = torch.split(
            side_embeds, [self.n_users, self.n_items], dim=0
        )
        content_embeds_user, content_embeds_items = torch.split(
            content_embeds, [self.n_users, self.n_items], dim=0
        )

        # projection된 item 임베딩
        h_mm_i_fusion = proj_item_dict["mm_proj"]  # (I,d)
        h_t_i_fusion = proj_item_dict["t_proj"]
        h_v_i_fusion = proj_item_dict["v_proj"]
        h_id_i_fusion = proj_item_dict["id_proj"]

        # graph-propagated raw item 임베딩
        mm_items_raw = raw_item_dict["mm_raw"]  # (I,d)
        t_items_raw = raw_item_dict["t_raw"]
        v_items_raw = raw_item_dict["v_raw"]

        # --- UI cosine alignment regularizer (user-anchor margin) ---
        if self.ui_cosine_loss:
            try:
                ui_w = float(self.ui_cosine_loss_weight)
            except Exception:
                if isinstance(self.ui_cosine_loss_weight, torch.Tensor):
                    ui_w = float(
                        self.ui_cosine_loss_weight.detach().cpu().item()
                    )
                else:
                    raise TypeError(
                        f"ui_cosine_loss_weight must be scalar-like, got {type(self.ui_cosine_loss_weight)}"
                    )

            u_anchor = u_g_embeddings.detach()  # (B,d)

            pos_align = 1 - F.cosine_similarity(
                u_anchor, pos_i_g_embeddings, dim=-1
            )
            neg_align = 1 - F.cosine_similarity(
                u_anchor, neg_i_g_embeddings, dim=-1
            )

            margin = 0.2
            uia_detached = F.relu(margin + pos_align - neg_align).mean()
            batch_mf_loss = batch_mf_loss + ui_w * uia_detached

        # --- REG Loss (아이템 구조 유지) ---
        t_embeds_user, t_embeds_items = torch.split(
            t_embeds, [self.n_users, self.n_items], dim=0
        )
        v_embeds_user, v_embeds_items = torch.split(
            v_embeds, [self.n_users, self.n_items], dim=0
        )

        pos_tt_batch_sim_mat = build_sim(self.t_feat[pos_items])
        pos_vv_batch_sim_mat = build_sim(self.v_feat[pos_items])
        pos_ii_batch_sim_mat = build_sim(self.mm_feat[pos_items])
        neg_ii_batch_sim_mat = build_sim(self.mm_feat[neg_items])

        if self.use_bce:
            ii_sim_loss = self.sim_sigmoid_loss(
                side_embeds_items[pos_items], pos_ii_batch_sim_mat
            ) + self.sim_sigmoid_loss(
                side_embeds_items[neg_items], neg_ii_batch_sim_mat
            )
            ii_t_sim_loss = self.sim_sigmoid_loss(
                t_embeds_items[pos_items], pos_tt_batch_sim_mat
            )
            ii_v_sim_loss = self.sim_sigmoid_loss(
                v_embeds_items[pos_items], pos_vv_batch_sim_mat
            )
        else:
            ii_sim_loss = self.sim_loss(
                side_embeds_items[pos_items], pos_ii_batch_sim_mat
            ) + 0.0 * self.sim_loss(
                side_embeds_items[neg_items], neg_ii_batch_sim_mat
            )
            ii_t_sim_loss = self.sim_loss(
                t_embeds_items[pos_items], pos_tt_batch_sim_mat
            )
            ii_v_sim_loss = self.sim_loss(
                v_embeds_items[pos_items], pos_vv_batch_sim_mat
            )

        ii_sim_loss = ii_sim_loss + 0.01 * ii_t_sim_loss + 0.01 * ii_v_sim_loss

        # --- 기존 cross-modal CL (InfoNCE) ---
        cl_id = self.InfoNCE(
            h_mm_i_fusion[pos_items],
            h_id_i_fusion[pos_items],
            0.2,
        )
        cl_v = self.InfoNCE(
            h_mm_i_fusion[pos_items].detach(),
            h_v_i_fusion[pos_items],
            0.2,
        )
        cl_t = self.InfoNCE(
            h_mm_i_fusion[pos_items].detach(),
            h_t_i_fusion[pos_items],
            0.2,
        )

        cl_loss = cl_id + self.lambda_weight * (cl_v + cl_t)

        # --- NEW: AI-MM / AI-S graded contrastive loss ---
        # 1) 배치 내 유니크 아이템
        batch_items = torch.unique(
            torch.cat([pos_items, neg_items], dim=0)
        )  # (B_i,)

        # 2) raw 임베딩으로 sim matrix 만들기 (멀티모달/텍스트)
        mm_batch_raw = mm_items_raw[batch_items]  # (B_i,d)
        t_batch_raw = t_items_raw[batch_items]  # (B_i,d)

        mm_sim = self._cosine_matrix(mm_batch_raw)  # (B_i,B_i)
        t_sim = self._cosine_matrix(t_batch_raw)

        # 3) graded sets R/T/N
        R_sets, T_sets, N_sets = self._build_graded_sets(
            mm_sim, t_sim, K_modal=5
        )

        # 4) projection된 임베딩에서 contrastive 계산
        mm_batch_proj = h_mm_i_fusion[batch_items]  # (B_i,d)
        t_batch_proj = h_t_i_fusion[batch_items]
        v_batch_proj = h_v_i_fusion[batch_items]

        ai_mm_mm, ai_s_mm = self._graded_contrastive_loss(
            mm_batch_proj, R_sets, T_sets, N_sets, self.tau
        )
        ai_mm_t, ai_s_t = self._graded_contrastive_loss(
            t_batch_proj, R_sets, T_sets, N_sets, self.tau
        )
        ai_mm_v, ai_s_v = self._graded_contrastive_loss(
            v_batch_proj, R_sets, T_sets, N_sets, self.tau
        )

        loss_ai_mm = (ai_mm_mm + ai_mm_t + ai_mm_v) / 3.0
        loss_ai_s = (ai_s_mm + ai_s_t + ai_s_v) / 3.0

        # --- base_loss (UI 안학습일 때도 쓰는 파트) ---
        base_loss = (
            batch_emb_loss
            + self.cl_loss * cl_loss
            + self.sim_weight * ii_sim_loss
            + self.gamma_mm * loss_ai_mm
            + self.gamma_s * loss_ai_s
        )

        if not_train_ui:
            return base_loss

        # 최종
        total_loss = batch_mf_loss + base_loss
        return total_loss

    # ======================
    # Inference
    # ======================

    def full_sort_predict(self, interaction):
        user = interaction[0]

        (
            restore_user_e,
            restore_item_e,
            mm_embeds,
            content_embeds,
            t_embeds,
            v_embeds,
        ) = self.forward(self.norm_adj, train=False)

        save_dir = "saved_emb/epoch_last"
        os.makedirs(save_dir, exist_ok=True)

        # RAW id-view item emb (LightGCN item part)
        h_id_raw = content_embeds[self.n_users :]  # (I,d)
        np.save(
            os.path.join(save_dir, "item_emb_raw_id.npy"),
            h_id_raw.detach().cpu().numpy(),
        )

        # raw text / vision feats
        np.save(
            os.path.join(save_dir, "item_feat_raw_text.npy"),
            self.t_feat.detach().cpu().float().numpy(),
        )
        np.save(
            os.path.join(save_dir, "item_feat_raw_vision.npy"),
            self.v_feat.detach().cpu().float().numpy(),
        )

        # final item emb actually used for scoring
        np.save(
            os.path.join(save_dir, "item_emb_final_alignrec.npy"),
            restore_item_e.detach().cpu().numpy(),
        )

        # multimodal before projection
        h_mm_out = mm_embeds[self.n_users :]
        np.save(
            os.path.join(save_dir, "item_emb_mm_out.npy"),
            h_mm_out.detach().cpu().numpy(),
        )

        # projection space snapshots
        with torch.no_grad():
            h_id_align = self.W_id_i(h_id_raw)
            h_mm_align = self.W_mm_i(h_mm_out)
            h_t_align = self.W_t(self.t_feat)
            h_v_align = self.W_v(self.v_feat)

            np.save(
                os.path.join(save_dir, "item_emb_align_id.npy"),
                h_id_align.cpu().numpy(),
            )
            np.save(
                os.path.join(save_dir, "item_emb_align_mm.npy"),
                h_mm_align.cpu().numpy(),
            )
            np.save(
                os.path.join(save_dir, "item_emb_align_text.npy"),
                h_t_align.cpu().numpy(),
            )
            np.save(
                os.path.join(save_dir, "item_emb_align_vision.npy"),
                h_v_align.cpu().numpy(),
            )

            h_proj_avg = (h_id_align + h_mm_align) / 2
            np.save(
                os.path.join(save_dir, "item_emb_align_projavg.npy"),
                h_proj_avg.cpu().numpy(),
            )

        # score = u dot i
        u_embeddings = restore_user_e[user]  # (d,)
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    # ======================
    # optional user hist init
    # ======================

    def init_user_hist_info(self, dataloader):
        uid_field = dataloader.dataset.uid_field
        iid_field = dataloader.dataset.iid_field
        time_field = "timestamp"

        uid_freq = dataloader.dataset.df.groupby(uid_field)[[iid_field, time_field]]
        result_dict = {
            uid: list(zip(group[iid_field], group[time_field]))
            for uid, group in uid_freq
        }
        for k in result_dict:
            result_dict[k] = sorted(
                result_dict[k], key=lambda tup: tup[1], reverse=True
            )

        topk = 10
        hist_topk = np.zeros((self.n_users, topk), dtype=np.int64) - 1
        for uid in result_dict:
            for i in range(min(topk, len(result_dict[uid]))):
                hist_topk[uid][topk - i - 1] = result_dict[uid][i][0]

        return torch.from_numpy(hist_topk).to(self.device)