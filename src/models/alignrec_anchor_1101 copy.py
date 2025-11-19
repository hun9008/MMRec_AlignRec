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
    # 제공되던 기반 Trainer 호환 클래스
from common.loss import EmbLoss
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph


class ALIGNREC_ANCHOR_1101(GeneralRecommender):
    def __init__(self, config, dataset):
        super(ALIGNREC_ANCHOR_1101, self).__init__(config, dataset)
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

        # 복원 디코더에서 나오는 ID-space 재주입 가중치
        self.recon_weight = config['recon_weight']

        # 멀티모달 마스킹 기반 self-supervised regularizer 관련 하이퍼
        # - mm_mask_ratio: mm_feat 차원 중 얼마나 죽일지 (0.3 등)
        # - mask_recon_weight: recon_mask_loss의 전체 loss 내 비중
        self.mm_mask_ratio = 0.3
        self.mask_recon_weight = 0.1

        # 최종 아이템 임베딩 만들 때 복원된 mm_rec_i를 섞을지 여부
        self.use_recovered_mm_for_final = True

        # dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # ID 임베딩 (user / item)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # ---------- Projection heads (to "alignment space") ----------
        # ID branch
        self.W_id_i = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        # MM branch
        self.W_mm_i = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        # Vision-only branch (raw v_feat -> proj)
        self.W_v = nn.Sequential(
            nn.Linear(self.v_feat.shape[1], self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        # Text-only branch (raw t_feat -> proj)
        self.W_t = nn.Sequential(
            nn.Linear(self.t_feat.shape[1], self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        # MM projection → ID space 복원 디코더
        # (alignment space의 h_mm_i_fusion을 다시 ID-style space로 역투영)
        self.D_mm2id = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        # init linear weights
        for block in [self.W_id_i, self.W_mm_i, self.W_v, self.W_t, self.D_mm2id]:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

        # ----- Graph structure 준비 -----
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(
            dataset_path,
            'mgcn_zkn_adj_{}_{}_{}.pt'.format(self.knn_k, self.sparse, self.desc)
        )

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        # ----- MM initial embedding + item-item graph adj -----
        if self.mm_feat is not None:
            self.mm_embedding = nn.Embedding.from_pretrained(self.mm_feat, freeze=False)
            mm_adj = build_sim(self.mm_embedding.weight.detach())
            mm_adj = build_knn_normalized_graph(
                mm_adj,
                topk=self.knn_k,
                is_sparse=self.sparse,
                norm_type='sym'
            )
            self.mm_original_adj = mm_adj.cuda()

            if self.use_ln:
                self.mm_ln = nn.LayerNorm(self.mm_feat.shape[1])
            self.mm_trs = nn.Linear(self.mm_feat.shape[1], self.embedding_dim)

        # ----- TEXT initial embedding + item-item graph adj -----
        if self.t_feat is not None:
            self.t_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            t_adj = build_sim(self.t_embedding.weight.detach())
            t_adj = build_knn_normalized_graph(
                t_adj,
                topk=self.knn_k,
                is_sparse=self.sparse,
                norm_type='sym'
            )
            self.t_original_adj = t_adj.cuda()

            if self.use_ln:
                self.t_ln = nn.LayerNorm(self.t_feat.shape[1])
            self.t_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        # ----- VISION initial embedding + item-item graph adj -----
        if self.v_feat is not None:
            self.v_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            v_adj = build_sim(self.v_embedding.weight.detach())
            v_adj = build_knn_normalized_graph(
                v_adj,
                topk=self.knn_k,
                is_sparse=self.sparse,
                norm_type='sym'
            )
            self.v_original_adj = v_adj.cuda()

            if self.use_ln:
                self.v_ln = nn.LayerNorm(self.v_feat.shape[1])
            self.v_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)

        # gates
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

        # sim_loss 옵션
        self.use_bce = False
        print("use_bce", self.use_bce)
        if self.use_bce:
            self.sigbce_loss = nn.BCEWithLogitsLoss()

        self.side_emb_div = config['side_emb_div']  # 기본 0

        # optional user history decoder stuff
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

    # -------------------------------------------------
    # Helper: bipartite 인접행렬 정규화
    # -------------------------------------------------
    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32
        )
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
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    # -------------------------------------------------
    # MM masking helpers (for self-supervised robustness)
    # -------------------------------------------------
    def get_masked_mm_views(self):
        """
        returns:
            mm_full   : (n_items, d_mm_raw)
            mm_masked : (n_items, d_mm_raw) with random feature dropout
        """
        mm_full = self.mm_embedding.weight  # learnable (n_items, raw_mm_dim)
        if self.mm_mask_ratio <= 0.0:
            return mm_full, mm_full

        # element-wise drop
        mask = (torch.rand_like(mm_full) > self.mm_mask_ratio).float()
        mm_masked = mm_full * mask
        return mm_full, mm_masked

    def mm_encode_items_only(self, mm_src_emb):
        """
        mm_src_emb: (n_items, raw_mm_dim) e.g. mm_full or mm_masked
        returns h_mm: (n_items, d) in alignment space after graph + W_mm_i

        This is a 'detached' mini-forward for item-side mm branch only.
        We DO NOT propagate to users here. We keep it item-only because
        this regularizer is meant to stabilize the item representation itself.
        """
        # project raw mm_src_emb into model dim via mm_trs (+ ln if use_ln)
        if self.use_ln:
            mm_item_embeds = self.mm_trs(self.mm_ln(mm_src_emb))
        else:
            mm_item_embeds = self.mm_trs(mm_src_emb)

        # gate with item_id_embedding to inject ID bias
        mm_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.gate_v(mm_item_embeds)
        )  # (n_items, d)

        # item-item graph propagation on MM view
        if self.sparse:
            for _ in range(self.n_layers):
                mm_item_embeds = torch.sparse.mm(self.mm_original_adj, mm_item_embeds)
        else:
            for _ in range(self.n_layers):
                mm_item_embeds = torch.mm(self.mm_original_adj, mm_item_embeds)

        # finally project to alignment space
        h_mm = self.W_mm_i(mm_item_embeds)  # (n_items, d)
        return h_mm

    # -------------------------------------------------
    # forward() = 메인 인퍼런스 경로 (LightGCN style + multimodal branches)
    # -------------------------------------------------
    def forward(self, adj, users=None, train=False):
        # ----- MM branch (per item) -----
        if self.mm_feat is not None:
            if self.use_ln:
                mm_feats = self.mm_trs(self.mm_ln(self.mm_embedding.weight))
            else:
                mm_feats = self.mm_trs(self.mm_embedding.weight)
        mm_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.gate_v(mm_feats)
        )

        # ----- TEXT branch -----
        if self.t_feat is not None:
            if self.use_ln:
                t_feats = self.t_trs(self.t_ln(self.t_embedding.weight))
            else:
                t_feats = self.t_trs(self.t_embedding.weight)
        t_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.gate_v(t_feats)
        )

        # ----- VISION branch -----
        if self.v_feat is not None:
            if self.use_ln:
                v_feats = self.v_trs(self.v_ln(self.v_embedding.weight))
            else:
                v_feats = self.v_trs(self.v_embedding.weight)
        v_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.gate_v(v_feats)
        )

        # ----- User-Item bipartite propagation (LightGCN style) -----
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

        # ----- item-item propagation for each modality -----
        # MM
        if self.sparse:
            for _ in range(self.n_layers):
                mm_item_embeds = torch.sparse.mm(self.mm_original_adj, mm_item_embeds)
        else:
            for _ in range(self.n_layers):
                mm_item_embeds = torch.mm(self.mm_original_adj, mm_item_embeds)
        mm_user_embeds = torch.sparse.mm(self.R, mm_item_embeds)
        mm_embeds = torch.cat([mm_user_embeds, mm_item_embeds], dim=0)

        # TEXT
        if self.sparse:
            for _ in range(self.n_layers):
                t_item_embeds = torch.sparse.mm(self.t_original_adj, t_item_embeds)
        else:
            for _ in range(self.n_layers):
                t_item_embeds = torch.mm(self.t_original_adj, t_item_embeds)
        t_user_embeds = torch.sparse.mm(self.R, t_item_embeds)
        t_embeds = torch.cat([t_user_embeds, t_item_embeds], dim=0)

        # VISION
        if self.sparse:
            for _ in range(self.n_layers):
                v_item_embeds = torch.sparse.mm(self.v_original_adj, v_item_embeds)
        else:
            for _ in range(self.n_layers):
                v_item_embeds = torch.mm(self.v_original_adj, v_item_embeds)
        v_user_embeds = torch.sparse.mm(self.R, v_item_embeds)
        v_embeds = torch.cat([v_user_embeds, v_item_embeds], dim=0)

        # ----- Split user/item embeddings -----
        content_embeds_user, content_embeds_items = torch.split(
            content_embeds, [self.n_users, self.n_items], dim=0
        )
        mm_embeds_user, mm_embeds_items = torch.split(
            mm_embeds, [self.n_users, self.n_items], dim=0
        )
        # (필요하면 t_embeds, v_embeds도 split 가능)

        # ----- Projection space (alignment heads) -----
        h_id_i_fusion = self.W_id_i(content_embeds_items)   # ID→proj
        h_mm_i_fusion = self.W_mm_i(mm_embeds_items)        # MM→proj

        # MM proj → ID-style 복원
        mm_rec_i = self.D_mm2id(h_mm_i_fusion)              # (n_items, d)

        # ----- Final embeddings for scoring -----
        if self.use_recovered_mm_for_final:
            # 기본 합: (ID graph) + (MM graph) + (TEXT graph)
            all_embeds = content_embeds + mm_embeds + t_embeds
            final_user_e, final_item_e = torch.split(
                all_embeds, [self.n_users, self.n_items], dim=0
            )
            # 복원된 mm_rec_i를 recover_lambda로 주입
            final_item_e = final_item_e + self.recover_lambda * mm_rec_i
        else:
            # vision까지 직접 더하는 버전
            all_embeds = content_embeds + mm_embeds + t_embeds + v_embeds
            final_user_e, final_item_e = torch.split(
                all_embeds, [self.n_users, self.n_items], dim=0
            )

        # optional user history decoder part
        if self.use_hist_decoder and train:
            hist_seq = mm_item_embeds[self.user_topk_hist[users], :]
            hist_seq = torch.where(
                torch.unsqueeze(self.user_topk_hist[users], dim=-1) == -1,
                torch.zeros_like(hist_seq),
                hist_seq
            )
            score = torch.bmm(
                self.query(hist_seq),
                self.key(hist_seq).transpose(1, 2)
            ) / np.sqrt(self.embedding_dim)
            score.masked_fill_(
                (self.user_topk_hist[users] == -1).view(-1, 1, 10),
                -float('Inf')
            )
            attn = F.softmax(score, -1)
            context = torch.bmm(attn, self.value(hist_seq))
            hist_hid = self.hist_ln1(hist_seq + context)

        if train:
            if self.use_hist_decoder:
                return (
                    final_user_e, final_item_e,
                    mm_embeds, content_embeds, t_embeds, v_embeds,
                    hist_hid[:, -1, :]
                )
            return (
                final_user_e, final_item_e,
                mm_embeds, content_embeds, t_embeds, v_embeds
            )

        # eval path
        return final_user_e, final_item_e, mm_embeds, content_embeds, t_embeds, v_embeds

    # -------------------------------------------------
    # core losses
    # -------------------------------------------------
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

        pos_score = (view1 * view2).sum(dim=-1)        # (B,)
        pos_score = torch.exp(pos_score / temperature) # (B,)

        ttl_score = torch.matmul(view1, view2.transpose(0, 1))  # (B,B)
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)  # (B,)

        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def sim_loss(self, embedding, sim):
        # embedding: (n_batch_items, d)
        # sim: (n_batch_items, n_batch_items) target sim matrix
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

    # -------------------------------------------------
    # training loss (per batch)
    # -------------------------------------------------
    def calculate_loss(self, interaction, not_train_ui=False):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        if self.use_hist_decoder:
            (
                ua_embeddings,
                ia_embeddings,
                side_embeds,
                content_embeds,
                t_embeds,
                v_embeds,
                user_hist_seq
            ) = self.forward(self.norm_adj, users, train=True)
        else:
            (
                ua_embeddings,
                ia_embeddings,
                side_embeds,
                content_embeds,
                t_embeds,
                v_embeds
            ) = self.forward(self.norm_adj, users, train=True)

        # ----- user/item batch view -----
        u_g_embeddings = ua_embeddings[users]          # (B,d)
        pos_i_g_embeddings = ia_embeddings[pos_items]  # (B,d)
        neg_i_g_embeddings = ia_embeddings[neg_items]  # (B,d)

        # ----- 기본 BPR loss -----
        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )

        # ----- global embeddings split -----
        side_embeds_users, side_embeds_items = torch.split(
            side_embeds, [self.n_users, self.n_items], dim=0
        )
        content_embeds_user, content_embeds_items = torch.split(
            content_embeds, [self.n_users, self.n_items], dim=0
        )

        # ----- projection (alignment space) -----
        h_id_i_fusion = self.W_id_i(content_embeds_items)   # (n_items,d)
        h_mm_i_fusion = self.W_mm_i(side_embeds_items)      # (n_items,d)

        # text / vision branch projection (direct from raw feats, not side_embeds_items)
        h_v_fusion = self.W_v(self.v_feat)                  # (n_items,d)
        h_t_fusion = self.W_t(self.t_feat)                  # (n_items,d)

        mixup_t_v_fusion = h_v_fusion * 0.25 + h_t_fusion * 0.75  # (n_items,d)

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

            # user branch detached (anchor)
            u_anchor = u_g_embeddings.detach()

            pos_align = 1 - F.cosine_similarity(u_anchor, pos_i_g_embeddings, dim=-1)
            neg_align = 1 - F.cosine_similarity(u_anchor, neg_i_g_embeddings, dim=-1)

            margin = 0.2
            uia_detached = F.relu(margin + pos_align - neg_align).mean()

            batch_mf_loss = batch_mf_loss + ui_w * uia_detached

        # ====================================================
        # 2) similarity regularizer (ii_sim_loss 등)
        # ====================================================
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
            )
            ii_t_sim_loss = self.sim_loss(
                t_embeds_items[pos_items], pos_tt_batch_sim_mat
            )
            ii_v_sim_loss = self.sim_loss(
                v_embeds_items[pos_items], pos_vv_batch_sim_mat
            )

        ii_sim_loss = ii_sim_loss + 0.03 * ii_t_sim_loss + 0.01 * ii_v_sim_loss

        # ====================================================
        # 3) Contrastive alignment losses
        #    - mm vs id (main)
        #    - mm vs text/vision/mixup (aux)
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

        # 여기서는 mm-id 직접 정렬(cl_id) + mixup 기반 정렬(cl_mixup)만 사용
        cl_loss = cl_id + cl_mixup

        # ====================================================
        # 4) 마스킹 기반 self-supervised reconstruction regularizer
        #     - 일부 feature drop 한 mm_masked 로부터 얻은 h_mm_masked
        #       가 full 정보로 만든 h_mm_full (detach) 를 복원하도록
        # ====================================================
        mm_full_raw, mm_masked_raw = self.get_masked_mm_views()
        h_mm_full = self.mm_encode_items_only(mm_full_raw)        # (n_items,d)
        h_mm_masked = self.mm_encode_items_only(mm_masked_raw)    # (n_items,d)

        # teacher-student 방식: full branch는 detach
        recon_mask_loss = F.mse_loss(
            h_mm_masked,
            h_mm_full.detach()
        )

        # ====================================================
        # 5) Loss 합산
        # ====================================================
        if not_train_ui:
            total_loss = (
                batch_emb_loss
                + cl_loss * self.cl_loss
                + self.sim_weight * ii_sim_loss
                + self.mask_recon_weight * recon_mask_loss
            )
            return total_loss

        total_loss = (
            batch_mf_loss
            + batch_emb_loss
            + cl_loss * self.cl_loss
            + self.sim_weight * ii_sim_loss
            + self.mask_recon_weight * recon_mask_loss
        )
        return total_loss

    # -------------------------------------------------
    # inference: full ranking scores
    # -------------------------------------------------
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

        # ---------- RAW ID-space item emb ----------
        h_id_raw = content_embeds[self.n_users:]  # (n_items,d)
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

        # ---------- final item emb after fusion ----------
        np.save(
            os.path.join(save_dir, "item_emb_final_alignrec.npy"),
            restore_item_e.detach().cpu().numpy()
        )

        h_mm_out = mm_embeds[self.n_users:]
        np.save(
            os.path.join(save_dir, "item_emb_mm_out.npy"),
            h_mm_out.detach().cpu().numpy()
        )

        # ---------- alignment-space dumps ----------
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

        u_embeddings = restore_user_e[user]  # (1,d) or (B,d)
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    # -------------------------------------------------
    # optional: top-k history encoder for users
    # -------------------------------------------------
    def init_user_hist_info(self, dataloader):
        uid_field = dataloader.dataset.uid_field
        iid_field = dataloader.dataset.iid_field
        time_field = 'timestamp'

        uid_freq = dataloader.dataset.df.groupby(uid_field)[iid_field, time_field]
        result_dict = {
            uid: list(zip(group[iid_field], group[time_field]))
            for uid, group in uid_freq
        }
        for k in result_dict:
            result_dict[k] = sorted(
                result_dict[k],
                key=lambda tup: tup[1],
                reverse=True
            )

        topk = 10
        hist_topk = np.zeros((self.n_users, topk), dtype=np.int) - 1
        # empty hist padded with -1
        for uid in result_dict:
            for i in range(min(topk, len(result_dict[uid]))):
                hist_topk[uid][topk - i - 1] = result_dict[uid][i][0]

        return torch.from_numpy(hist_topk).to(self.device)