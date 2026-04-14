import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss
from utils.utils import build_sim, build_knn_normalized_graph

class ANCHORREC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(ANCHORREC, self).__init__(config, dataset)
        self.sparse = True
        self.aal_loss = config['aal_loss'] # lambda_1
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.lambda_weight = config['lambda_weight']
        self.use_ln = config['use_ln']
        self.amp_loss = config['amp_loss'] # lambda_2
        self.ui_cosine_loss_weight = config['ui_cosine_loss_weight']
        self.reg_loss = EmbLoss()
        self.recover_lambda = config['recover_lambda']
        self.tau = config['tau_weight']
        self.mask_dropout = nn.Dropout(p=0.2)
        self.ui_cosine_loss = config['ui_cosine_loss']
        self.save_eval_embeddings = config['save_eval_embeddings'] if 'save_eval_embeddings' in config else False

        if self.mm_feat is None:
            raise ValueError(f"ANCHORREC requires mm_feat\n{config}")
        if self.t_feat is None:
            raise ValueError(f"ANCHORREC requires t_feat\n{config}")
        if self.v_feat is None:
            raise ValueError(f"ANCHORREC requires v_feat\n{config}")
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

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

        self.D_mm2id = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        for block in [self.W_id_i, self.W_mm_i, self.W_v, self.W_t, self.D_mm2id]:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        self.mm_embedding = nn.Embedding.from_pretrained(self.mm_feat, freeze=False)
        mm_adj = build_sim(self.mm_embedding.weight.detach())
        mm_adj = build_knn_normalized_graph(mm_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
        self.mm_original_adj = mm_adj.to(self.device)
        if not self.mm_original_adj.is_coalesced():
            self.mm_original_adj = self.mm_original_adj.coalesce()

        self.t_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        t_adj = build_sim(self.t_embedding.weight.detach())
        t_adj = build_knn_normalized_graph(t_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
        self.t_original_adj = t_adj.to(self.device)

        self.v_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        v_adj = build_sim(self.v_embedding.weight.detach())
        v_adj = build_knn_normalized_graph(v_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
        self.v_original_adj = v_adj.to(self.device)

        if self.use_ln:
            self.mm_ln = nn.LayerNorm(self.mm_feat.shape[1])
            self.t_ln = nn.LayerNorm(self.t_feat.shape[1])
            self.v_ln = nn.LayerNorm(self.v_feat.shape[1])
        self.mm_trs = nn.Linear(self.mm_feat.shape[1], self.embedding_dim)
        self.t_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.v_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)

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
            norm_adj = d_mat_inv.dot(adj)
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

    def _propagate_item_graph(self, adj, item_embeds):
        if self.sparse:
            for _ in range(self.n_layers):
                item_embeds = torch.sparse.mm(adj, item_embeds)
            return item_embeds
        for _ in range(self.n_layers):
            item_embeds = torch.mm(adj, item_embeds)
        return item_embeds

    def forward(self, adj, users=None, train=False):
        if self.use_ln:
            mm_feats = self.mm_trs(self.mm_ln(self.mm_embedding.weight))
            t_feats = self.t_trs(self.t_ln(self.t_embedding.weight))
            v_feats = self.v_trs(self.v_ln(self.v_embedding.weight))
        else:
            mm_feats = self.mm_trs(self.mm_embedding.weight)
            t_feats = self.t_trs(self.t_embedding.weight)
            v_feats = self.v_trs(self.v_embedding.weight)

        mm_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_mm(mm_feats))
        t_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(t_feats))
        v_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(v_feats))

        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        mm_item_embeds = self._propagate_item_graph(self.mm_original_adj, mm_item_embeds)
        mm_user_embeds = torch.sparse.mm(self.R, mm_item_embeds)
        mm_embeds = torch.cat([mm_user_embeds, mm_item_embeds], dim=0)

        t_item_embeds = self._propagate_item_graph(self.t_original_adj, t_item_embeds)
        t_user_embeds = torch.sparse.mm(self.R, t_item_embeds)
        t_embeds = torch.cat([t_user_embeds, t_item_embeds], dim=0)

        v_item_embeds = self._propagate_item_graph(self.v_original_adj, v_item_embeds)
        v_user_embeds = torch.sparse.mm(self.R, v_item_embeds)
        v_embeds = torch.cat([v_user_embeds, v_item_embeds], dim=0)

        _, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        _, mm_embeds_items = torch.split(mm_embeds, [self.n_users, self.n_items], dim=0)
        mm_rec_i = self.D_mm2id(self.W_mm_i(mm_embeds_items))

        all_embeds = content_embeds + mm_embeds + t_embeds
        final_user_e, final_item_e = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
        final_item_e = final_item_e + self.recover_lambda * mm_rec_i

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
        aal_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(aal_loss)
    
    def sim_loss(self, embedding, sim):
        embedding_sim = torch.mm(embedding, embedding.t())
        sim_loss = self.reg_loss(embedding_sim - sim.detach())
        return sim_loss

    def calculate_loss(self, interaction, not_train_ui=False):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds, t_embeds, v_embeds = \
                self.forward(self.norm_adj, users, train=True)

        _, t_embeds_items = torch.split(t_embeds, [self.n_users, self.n_items], dim=0)
        _, v_embeds_items = torch.split(v_embeds, [self.n_users, self.n_items], dim=0)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )

        _, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        _, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)

        h_id_i_fusion = self.W_id_i(content_embeds_items)
        h_mm_i_fusion = self.W_mm_i(side_embeds_items)

        h_v_fusion = self.W_v(self.v_feat)
        h_t_fusion = self.W_t(self.t_feat)

        mixup_t_v_fusion = h_v_fusion * 0.25 + h_t_fusion * 0.75

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

        pos_tt_batch_sim_mat = build_sim(self.t_feat[pos_items])
        pos_vv_batch_sim_mat = build_sim(self.v_feat[pos_items])
        pos_ii_batch_sim_mat = build_sim(self.mm_feat[pos_items])
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

        cl_id = self.InfoNCE(
            h_mm_i_fusion[pos_items],
            h_id_i_fusion[pos_items],
            self.tau
        )
        cl_mixup = self.InfoNCE(
            h_mm_i_fusion[pos_items].detach(),
            mixup_t_v_fusion[pos_items],
            self.tau
        )
        aal_loss = cl_id + cl_mixup

        row_idx, col_idx = self.mm_original_adj.indices()
        w_ij = self.mm_original_adj.values()

        zi = side_embeds_items[row_idx]
        zj = side_embeds_items[col_idx]
        graph_smooth_loss = ((zi - zj) ** 2).sum(dim=1) * w_ij
        graph_smooth_loss = graph_smooth_loss.mean()
        batch_reg_loss = graph_smooth_loss

        if not_train_ui:
            total_loss = (
                batch_emb_loss
                + aal_loss * self.aal_loss
                + self.amp_loss * ii_sim_loss
                + 0.1 * batch_reg_loss
            )
            return total_loss

        total_loss = (
            batch_mf_loss
            + batch_emb_loss
            + aal_loss * self.aal_loss
            + self.amp_loss * ii_sim_loss
            + 0.1 * batch_reg_loss
        )
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        (
            restore_user_e,
            restore_item_e,
            mm_embeds,
            content_embeds,
            _,
            _
        ) = self.forward(self.norm_adj)

        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def init_user_hist_info(self, dataloader):
        uid_field = dataloader.dataset.uid_field
        iid_field = dataloader.dataset.iid_field
        time_field = 'timestamp'
        # load avail items for all uid
        uid_freq = dataloader.dataset.df.groupby(uid_field)[iid_field,time_field]
        result_dict = {uid: list(zip(group[iid_field], group[time_field])) for uid, group in uid_freq}
        for k in result_dict:
            result_dict[k] = sorted(result_dict[k], key=lambda tup: tup[1],reverse=True)
        topk = 10
        hist_topk = np.zeros((self.n_users,topk),dtype=np.int)-1
        # empty hist is padded with -1 ''before'' the hist sequence
        for uid in result_dict:
            for i in range(min(topk,len(result_dict[k]))):
                hist_topk[uid][topk-i-1] = result_dict[uid][i][0]
        return torch.from_numpy(hist_topk).to(self.device)
