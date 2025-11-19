import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_max
except:
    pass

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph


class ALIGNREC_ANCHOR_1103_LINE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(ALIGNREC_ANCHOR_1103_LINE, self).__init__(config, dataset)
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
        self.use_cross_att = False
        self.use_user_history = False
        self.add_user_history_after_content_embs = False
        self.reg_loss = EmbLoss()
        self.recover_lambda = config['recover_lambda']

        # ================= NEW: modality-attention hyperparams (no config.get) =================
        self.att_d_k = self.embedding_dim            # key/query dim
        self.att_use_ln = True                       # use LayerNorm
        self.att_loss_weight = 0.1                   # consistency loss weight
        self.att_cl_weight = 0.1                     # CL weight for attention outputs
        self.att_fuse_lambda = 0.5                   # residual fuse ratio into final_item_e

        # layers for modality attention (id as Q; mm/t/v as K,V)
        self.att_q = nn.Linear(self.embedding_dim, self.att_d_k, bias=False)
        self.att_k = nn.Linear(self.embedding_dim, self.att_d_k, bias=False)
        self.att_v = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        if self.att_use_ln:
            self.att_ln_q = nn.LayerNorm(self.att_d_k)
            self.att_ln_k = nn.LayerNorm(self.att_d_k)
            self.att_ln_v = nn.LayerNorm(self.embedding_dim)

        self.mask_dropout = nn.Dropout(p=0.2)

        # 새로 추가: recon loss 가중치 (gamma 같은 역할) – 사용하지 않더라도 보존
        self.recon_weight = config['recon_weight']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # === Projection Weights (alignment space) ===
        self.W_id_i = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.W_mm_i = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.W_v = nn.Sequential(
            nn.Linear(self.v_feat.shape[1], self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.W_t = nn.Sequential(
            nn.Linear(self.t_feat.shape[1], self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        # ★ 복원 디코더: projection-space(mm) → ID-space
        self.D_mm2id = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        # init weights
        for block in [self.W_id_i, self.W_mm_i, self.W_v, self.W_t, self.D_mm2id, self.att_q, self.att_k, self.att_v]:
            if isinstance(block, nn.Sequential):
                for layer in block:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
            elif isinstance(block, nn.Linear):
                nn.init.xavier_uniform_(block.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, f'mgcn_zkn_adj_{self.knn_k}_{self.sparse}_{self.desc}.pt')

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.mm_feat is not None:
            self.mm_embedding = nn.Embedding.from_pretrained(self.mm_feat, freeze=False)
            mm_adj = build_sim(self.mm_embedding.weight.detach())
            mm_adj = build_knn_normalized_graph(mm_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
            self.mm_original_adj = mm_adj.cuda()
            if self.mm_original_adj.is_coalesced() is False:
                self.mm_original_adj = self.mm_original_adj.coalesce()

        if self.mm_feat is not None:
            if self.use_ln:
                self.mm_ln = nn.LayerNorm(self.mm_feat.shape[1])
            self.mm_trs = nn.Linear(self.mm_feat.shape[1], self.embedding_dim)

        if self.t_feat is not None:
            self.t_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            t_adj = build_sim(self.t_embedding.weight.detach())
            t_adj = build_knn_normalized_graph(t_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
            self.t_original_adj = t_adj.cuda()

        if self.t_feat is not None:
            if self.use_ln:
                self.t_ln = nn.LayerNorm(self.t_feat.shape[1])
            self.t_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        if self.v_feat is not None:
            self.v_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            v_adj = build_sim(self.v_embedding.weight.detach())
            v_adj = build_knn_normalized_graph(v_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
            self.v_original_adj = v_adj.cuda()

        if self.v_feat is not None:
            if self.use_ln:
                self.v_ln = nn.LayerNorm(self.v_feat.shape[1])
            self.v_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.gate_mm_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_mm = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.use_bce = False
        print("use_bce", self.use_bce)
        if self.use_bce:
            self.sigbce_loss = nn.BCEWithLogitsLoss()

        self.side_emb_div = config['side_emb_div']  # set to 0 by default

        self.use_hist_decoder = config['use_hist_decoder']  # False
        if self.use_hist_decoder:
            self.user_topk_hist = self.init_user_hist_info(dataset)
            self.query = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.key = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.value = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.hist_ln1 = nn.LayerNorm(self.embedding_dim)

        self.test_arch1 = config['test_arch1']  # False
        if self.test_arch1:
            self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
            nn.init.xavier_normal_(self.predictor.weight)

        self.ui_cosine_loss = config['ui_cosine_loss']  # False

        self.tau = 0.5

    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, users=None, train=False):
        # ----- MM features (projected) -----
        if self.mm_feat is not None:
            if self.use_ln:
                mm_feats = self.mm_trs(self.mm_ln(self.mm_embedding.weight))
            else:
                mm_feats = self.mm_trs(self.mm_embedding.weight)
        mm_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_mm(mm_feats))

        # ----- Text -----
        if self.t_feat is not None:
            if self.use_ln:
                t_feats = self.t_trs(self.t_ln(self.t_embedding.weight))
            else:
                t_feats = self.t_trs(self.t_embedding.weight)
        t_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(t_feats))

        # ----- Vision -----
        if self.v_feat is not None:
            if self.use_ln:
                v_feats = self.v_trs(self.v_ln(self.v_embedding.weight))
            else:
                v_feats = self.v_trs(self.v_embedding.weight)
        v_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(v_feats))

        # ----- UI view (LightGCN-like propagation over user-item bipartite) -----
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1, keepdim=False)
        content_embeds = all_embeddings  # (n_users+n_items, d)

        # ----- Graph propagation in item-item MM view -----
        if self.sparse:
            for _ in range(self.n_layers):
                mm_item_embeds = torch.sparse.mm(self.mm_original_adj, mm_item_embeds)
        else:
            for _ in range(self.n_layers):
                mm_item_embeds = torch.mm(self.mm_original_adj, mm_item_embeds)
        mm_user_embeds = torch.sparse.mm(self.R, mm_item_embeds)
        mm_embeds = torch.cat([mm_user_embeds, mm_item_embeds], dim=0)

        # ----- Graph propagation in item-item TEXT view -----
        if self.sparse:
            for _ in range(self.n_layers):
                t_item_embeds = torch.sparse.mm(self.t_original_adj, t_item_embeds)
        else:
            for _ in range(self.n_layers):
                t_item_embeds = torch.mm(self.t_original_adj, t_item_embeds)
        t_user_embeds = torch.sparse.mm(self.R, t_item_embeds)
        t_embeds = torch.cat([t_user_embeds, t_item_embeds], dim=0)

        # ----- Graph propagation in item-item VISION view -----
        if self.sparse:
            for _ in range(self.n_layers):
                v_item_embeds = torch.sparse.mm(self.v_original_adj, v_item_embeds)
        else:
            for _ in range(self.n_layers):
                v_item_embeds = torch.mm(self.v_original_adj, v_item_embeds)
        v_user_embeds = torch.sparse.mm(self.R, v_item_embeds)
        v_embeds = torch.cat([v_user_embeds, v_item_embeds], dim=0)

        # ----- Split user/item -----
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        mm_embeds_user,     mm_embeds_items     = torch.split(mm_embeds,     [self.n_users, self.n_items], dim=0)
        t_embeds_user,      t_embeds_items      = torch.split(t_embeds,      [self.n_users, self.n_items], dim=0)
        v_embeds_user,      v_embeds_items      = torch.split(v_embeds,      [self.n_users, self.n_items], dim=0)

        # ===== (NEW) Projection 전 모달리티-어텐션 (id as Q; mm/t/v as K,V) =====
        id_q = self.att_q(content_embeds_items)                     # (N_i, d_k)
        mm_k = self.att_k(mm_embeds_items)                          # (N_i, d_k)
        t_k  = self.att_k(t_embeds_items)
        v_k  = self.att_k(v_embeds_items)

        mm_v = self.att_v(mm_embeds_items)                          # (N_i, d)
        t_v  = self.att_v(t_embeds_items)
        v_v  = self.att_v(v_embeds_items)

        if self.att_use_ln:
            id_q = self.att_ln_q(id_q)
            mm_k = self.att_ln_k(mm_k)
            t_k  = self.att_ln_k(t_k)
            v_k  = self.att_ln_k(v_k)
            mm_v = self.att_ln_v(mm_v)
            t_v  = self.att_ln_v(t_v)
            v_v  = self.att_ln_v(v_v)

        K = torch.stack([mm_k, t_k, v_k], dim=1)                   # (N_i, 3, d_k)
        V = torch.stack([mm_v, t_v, v_v], dim=1)                   # (N_i, 3, d)
        att_scores = torch.einsum('nd,nmd->nm', id_q, K) / (self.att_d_k ** 0.5)  # (N_i, 3)
        att_w = F.softmax(att_scores, dim=1)
        att_out = torch.einsum('nm,nmd->nd', att_w, V)             # (N_i, d)

        # 저장하여 loss/분석에서 활용
        self.preproj_att_out = att_out
        self.preproj_id_raw  = content_embeds_items
        self.preproj_mm_raw  = mm_embeds_items
        self.preproj_t_raw   = t_embeds_items
        self.preproj_v_raw   = v_embeds_items
        self.preproj_att_w   = att_w

        # ----- Projection space (alignment) -----
        h_id_i_fusion = self.W_id_i(content_embeds_items)     # (n_items, d)
        h_mm_i_fusion = self.W_mm_i(mm_embeds_items)          # (n_items, d)

        # MM proj → ID space 복원
        mm_rec_i = self.D_mm2id(h_mm_i_fusion)                # (n_items, d)

        # ----- Final embeddings for scoring -----
        all_embeds = content_embeds + mm_embeds + t_embeds
        final_user_e, final_item_e = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        # (NEW) attention residual
        final_item_e = final_item_e + self.att_fuse_lambda * att_out
        # 기존 복원 잔차
        final_item_e = final_item_e + self.recover_lambda * mm_rec_i

        if train:
            return final_user_e, final_item_e, mm_embeds, content_embeds, t_embeds, v_embeds, att_out
        # eval 경로
        return final_user_e, final_item_e, mm_embeds, content_embeds, t_embeds, v_embeds

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        regularizer = 1. / 2 * (users ** 2).sum() \
                    + 1. / 2 * (pos_items ** 2).sum() \
                    + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        emb_loss = self.lambda_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def sim_loss(self, embedding, sim):
        embedding_sim = torch.mm(embedding, embedding.t())
        sim_loss = self.reg_loss(embedding_sim - sim.detach())
        return sim_loss

    def sim_sigmoid_loss(self, embedding, sim):
        embedding_sim = torch.mm(embedding, embedding.t())
        logit_emb_sim = torch.reshape(embedding_sim, (-1, 1))
        logit_sim = torch.reshape(sim, (-1, 1))
        target = torch.sigmoid(logit_sim)
        sim_loss = self.sigbce_loss(logit_emb_sim, target)
        return sim_loss

    def _neg_cosine(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return -(x * y).sum(dim=1).mean()

    def _intra_mask_loss_identity(self, h):
        online = F.normalize(h, dim=1)
        with torch.no_grad():
            target = F.normalize(self.mask_dropout(h), dim=1)
        return -(online * target).sum(dim=1).mean()

    def calculate_loss(self, interaction, not_train_ui=False):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds, t_embeds, v_embeds, att_out = \
                self.forward(self.norm_adj, users, train=True)

        t_embeds_user, t_embeds_items = torch.split(t_embeds, [self.n_users, self.n_items], dim=0)
        v_embeds_user, v_embeds_items = torch.split(v_embeds, [self.n_users, self.n_items], dim=0)

        # user/item batch view
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        # 기본 BPR loss
        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )

        # split global embeddings back to user/item
        side_embeds_users, side_embeds_items = torch.split(
            side_embeds, [self.n_users, self.n_items], dim=0
        )
        content_embeds_user, content_embeds_items = torch.split(
            content_embeds, [self.n_users, self.n_items], dim=0
        )

        # projection (alignment space)
        h_id_i_fusion = self.W_id_i(content_embeds_items)   # (n_items, d)
        h_mm_i_fusion = self.W_mm_i(side_embeds_items)      # (n_items, d)

        # text / vision projection
        h_v_fusion = self.W_v(self.v_feat)                  # (n_items, d)
        h_t_fusion = self.W_t(self.t_feat)                  # (n_items, d)

        mixup_t_v_fusion = h_v_fusion * 0.25 + h_t_fusion * 0.75

        # ====================================================
        # 1) UI cosine margin ranking regularizer (optional)
        # ====================================================
        if self.ui_cosine_loss:
            try:
                ui_w = float(self.ui_cosine_loss_weight)
            except Exception:
                if isinstance(self.ui_cosine_loss_weight, torch.Tensor):
                    ui_w = float(self.ui_cosine_loss_weight.detach().cpu().item())
                else:
                    raise TypeError(
                        f"ui_cosine_loss_weight must be a float-like scalar, got {type(self.ui_cosine_loss_weight)}"
                    )
            u_anchor = u_g_embeddings.detach()
            pos_align = 1 - F.cosine_similarity(u_anchor, pos_i_g_embeddings, dim=-1)
            neg_align = 1 - F.cosine_similarity(u_anchor, neg_i_g_embeddings, dim=-1)
            margin = 0.2
            uia_detached = F.relu(margin + pos_align - neg_align).mean()
            batch_mf_loss = batch_mf_loss + ui_w * uia_detached

        # ====================================================
        # 2) similarity regularizer (ii_sim_loss 등) REG Loss
        # ====================================================
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
            )
            ii_t_sim_loss = self.sim_loss(
                t_embeds_items[pos_items], pos_tt_batch_sim_mat
            )
            ii_v_sim_loss = self.sim_loss(
                v_embeds_items[pos_items], pos_vv_batch_sim_mat
            )
        ii_sim_loss = ii_sim_loss + 0.03 * ii_t_sim_loss + 0.01 * ii_v_sim_loss

        # ====================================================
        # 3) Contrastive alignment losses (projection space)
        # ====================================================
        cl_id = self.InfoNCE(
            h_mm_i_fusion[pos_items],
            h_id_i_fusion[pos_items],
            0.2
        )
        cl_v = self.InfoNCE(
            h_mm_i_fusion[pos_items].detach(),
            h_v_fusion[pos_items],
            0.2
        )
        cl_t = self.InfoNCE(
            h_mm_i_fusion[pos_items].detach(),
            h_t_fusion[pos_items],
            0.2
        )
        cl_mixup = self.InfoNCE(
            h_mm_i_fusion[pos_items].detach(),
            mixup_t_v_fusion[pos_items],
            0.2
        )
        cl_loss = cl_id + cl_mixup

        # ====================================================
        # 4) Graph smoothness reg on mm graph
        # ====================================================
        row_idx, col_idx = self.mm_original_adj.indices()
        w_ij = self.mm_original_adj.values()
        zi = side_embeds_items[row_idx]
        zj = side_embeds_items[col_idx]
        graph_smooth_loss = ((zi - zj) ** 2).sum(dim=1) * w_ij
        graph_smooth_loss = graph_smooth_loss.mean()
        batch_reg_loss = graph_smooth_loss

        # ====================================================
        # 5) (NEW) Attention auxiliary losses
        #   - Consistency: att_out ≈ id_raw (cosine)
        #   - CL: att_out vs mm_raw, and att_out vs (0.25*t + 0.75*v)
        # ====================================================
        id_raw = self.preproj_id_raw
        mm_raw = self.preproj_mm_raw
        t_raw  = self.preproj_t_raw
        v_raw  = self.preproj_v_raw
        att_raw = att_out

        att_consistency = 1.0 - F.cosine_similarity(id_raw[pos_items], att_raw[pos_items], dim=-1).mean()
        att_cl_mm = self.InfoNCE(att_raw[pos_items], mm_raw[pos_items].detach(), 0.2)
        tv_mix = 0.25 * t_raw + 0.75 * v_raw
        att_cl_tv = self.InfoNCE(att_raw[pos_items], tv_mix[pos_items].detach(), 0.2)
        att_loss = self.att_loss_weight * att_consistency + self.att_cl_weight * (att_cl_mm + att_cl_tv)

        # ====================================================
        # 6) Final loss combine
        # ====================================================
        if not_train_ui:
            total_loss = (
                batch_emb_loss
                + cl_loss * self.cl_loss
                + self.sim_weight * ii_sim_loss
                + 0.1 * batch_reg_loss
                + att_loss
            )
            return total_loss

        total_loss = (
            batch_mf_loss
            + batch_emb_loss
            + cl_loss * self.cl_loss
            + self.sim_weight * ii_sim_loss
            + 0.1 * batch_reg_loss
            + att_loss
        )
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        (
            restore_user_e,
            restore_item_e,
            mm_embeds,
            content_embeds,
            t_embeds,
            v_embeds
        ) = self.forward(self.norm_adj)

        save_dir = "saved_emb/epoch_last"
        os.makedirs(save_dir, exist_ok=True)

        # ---------- Alignment 미반영 (RAW) ----------
        h_id_raw = content_embeds[self.n_users:]  # (n_items, d)
        np.save(
            os.path.join(save_dir, "item_emb_raw_id.npy"),
            h_id_raw.detach().cpu().numpy()
        )
        np.save(
            os.path.join(save_dir, "item_feat_raw_text.npy"),
            self.t_feat.detach().cpu().float().numpy()
        )
        np.save(
            os.path.join(save_dir, "item_feat_raw_vision.npy"),
            self.v_feat.detach().cpu().float().numpy()
        )

        # ---------- Alignment 반영 ----------
        np.save(
            os.path.join(save_dir, "item_emb_final_alignrec.npy"),
            restore_item_e.detach().cpu().numpy()
        )
        h_mm_out = mm_embeds[self.n_users:]
        np.save(
            os.path.join(save_dir, "item_emb_mm_out.npy"),
            h_mm_out.detach().cpu().numpy()
        )

        with torch.no_grad():
            h_id_align = self.W_id_i(h_id_raw)
            h_mm_align = self.W_mm_i(h_mm_out)
            h_t_align  = self.W_t(self.t_feat)
            h_v_align  = self.W_v(self.v_feat)
            np.save(
                os.path.join(save_dir, "item_emb_align_id.npy"),
                h_id_align.cpu().numpy()
            )
            np.save(
                os.path.join(save_dir, "item_emb_align_mm.npy"),
                h_mm_align.cpu().numpy()
            )
            np.save(
                os.path.join(save_dir, "item_emb_align_text.npy"),
                h_t_align.cpu().numpy()
            )
            np.save(
                os.path.join(save_dir, "item_emb_align_vision.npy"),
                h_v_align.cpu().numpy()
            )
            h_proj_avg = (h_id_align + h_mm_align) / 2
            np.save(
                os.path.join(save_dir, "item_emb_align_projavg.npy"),
                h_proj_avg.cpu().numpy()
            )

        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def init_user_hist_info(self, dataloader):
        uid_field = dataloader.dataset.uid_field
        iid_field = dataloader.dataset.iid_field
        time_field = 'timestamp'
        uid_freq = dataloader.dataset.df.groupby(uid_field)[iid_field, time_field]
        result_dict = {uid: list(zip(group[iid_field], group[time_field])) for uid, group in uid_freq}
        for k in result_dict:
            result_dict[k] = sorted(result_dict[k], key=lambda tup: tup[1], reverse=True)
        topk = 10
        hist_topk = np.zeros((self.n_users, topk), dtype=np.int32) - 1
        # empty hist is padded with -1 before the hist sequence
        for uid in result_dict:
            for i in range(min(topk, len(result_dict[uid]))):
                hist_topk[uid][topk - i - 1] = result_dict[uid][i][0]
        return torch.from_numpy(hist_topk).to(self.device)
