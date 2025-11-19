# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
LATTICE
################################################
Reference:
    https://github.com/CRIPAC-DIG/LATTICE
    ACM MM'2021: [Mining Latent Structures for Multimedia Recommendation]
    https://arxiv.org/abs/2104.09036
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
# from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood  # ❌ 사용 안함

# =========================
# Sparse helpers (OOM-safe)
# =========================
@torch.no_grad()
def sparse_normalized_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalized adjacency: D^{-1/2} A D^{-1/2}
    입력 adj는 sparse(coo) 또는 dense 가능. 결과는 sparse coo 반환.
    """
    # dense -> sparse 변환 (dense 거대 행렬은 상위에서 만들지 않지만 방어적으로 처리)
    if not adj.is_sparse:
        idx = adj.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return adj
        vals = adj[idx[:, 0], idx[:, 1]]
        adj = torch.sparse_coo_tensor(idx.t(), vals, adj.shape, device=adj.device)

    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    row, col = indices[0], indices[1]
    n = adj.size(0)

    deg = torch.zeros(n, device=values.device, dtype=values.dtype)
    deg.index_add_(0, row, values)
    deg = torch.clamp(deg, min=1e-12)
    d_inv_sqrt = deg.pow(-0.5)

    norm_values = d_inv_sqrt[row] * values * d_inv_sqrt[col]
    norm_adj = torch.sparse_coo_tensor(indices, norm_values, adj.shape, device=adj.device)
    return norm_adj.coalesce()

@torch.no_grad()
def build_knn_sparse_from_feats(feats: torch.Tensor, k: int, device: torch.device, row_bs: int = 1024) -> torch.Tensor:
    """
    features (n_items, d) 로부터 **dense NxN을 만들지 않고** top-k KNN sparse 그래프 생성.
    - cosine 유사도 (L2 normalize 후 내적)
    - 자기자신은 제외
    - 반환: sparse coo (shape: n x n)
    """
    n, d = feats.shape
    # 정규화 후 원하는 device로 이동
    x = F.normalize(feats, p=2, dim=1).to(device, non_blocking=True)
    xt = x.t().contiguous()

    row_list = []
    col_list = []
    val_list = []

    for start in range(0, n, row_bs):
        end = min(start + row_bs, n)
        xb = x[start:end]                      # (b, d)
        sims = xb @ xt                         # (b, n), dense 한 블록이지만 b*n만큼만 유지
        # 자기자신 제외(해당 구간 대각선만)
        b = end - start
        ar = torch.arange(b, device=device)
        sims[ar, start:end] = -float('inf')

        # k+1 대비 안전, 하지만 self를 뺐으니 k로 충분
        topv, topi = torch.topk(sims, k, dim=1, largest=True, sorted=False)  # (b, k)

        # 수집
        row_idx = (ar.unsqueeze(1) + start).expand_as(topi)                  # (b, k)
        row_list.append(row_idx.reshape(-1))
        col_list.append(topi.reshape(-1))
        val_list.append(topv.reshape(-1))

        # 메모리 해제 힌트
        del xb, sims, topv, topi

    rows = torch.cat(row_list, dim=0)
    cols = torch.cat(col_list, dim=0)
    vals = torch.cat(val_list, dim=0)
    # 음수 무한대 방지(이론상 없음), 음수도 허용하지만 0 이하는 cut 할 수도 있음
    mask = torch.isfinite(vals)
    rows, cols, vals = rows[mask], cols[mask], vals[mask]

    adj = torch.sparse_coo_tensor(torch.stack([rows, cols], dim=0), vals, (n, n), device=device)
    return adj.coalesce()

class LATTICE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LATTICE, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.weight_size = config['weight_size']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_adj_mat()
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
        self.item_adj = None

        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if config['cf_model'] == 'ngcf':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            dropout_list = config['mess_dropout']
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.dropout_list.append(nn.Dropout(dropout_list[i]))

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, f'image_adj_{self.knn_k}.pt')
        text_adj_file  = os.path.join(dataset_path, f'text_adj_{self.knn_k}.pt')

        # 멀티모달 임베딩 레이어
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        if self.t_feat is not None:
            self.text_embedding  = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        # KNN adj (원본) 로드/생성: 항상 sparse 정규화본으로 유지
        if self.v_feat is not None:
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file, map_location=self.device)
                # dense로 저장되어 있어도 안전 변환
                image_adj = sparse_normalized_adj(image_adj).to(self.device)
            else:
                # features -> proj -> sparse knn -> normalize
                with torch.no_grad():
                    img_feats = self.image_embedding.weight.detach()
                    if img_feats.device != self.device:
                        img_feats = img_feats.to(self.device)
                    image_adj = build_knn_sparse_from_feats(img_feats, self.knn_k, self.device, row_bs=1024)
                    image_adj = sparse_normalized_adj(image_adj)
                    torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj

        if self.t_feat is not None:
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file, map_location=self.device)
                text_adj = sparse_normalized_adj(text_adj).to(self.device)
            else:
                with torch.no_grad():
                    txt_feats = self.text_embedding.weight.detach()
                    if txt_feats.device != self.device:
                        txt_feats = txt_feats.to(self.device)
                    text_adj = build_knn_sparse_from_feats(txt_feats, self.knn_k, self.device, row_bs=1024)
                    text_adj  = sparse_normalized_adj(text_adj)
                    torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj

        # feature projector
        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_trs  = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)

    def pre_epoch_processing(self):
        self.build_item_graph = True

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values  = torch.from_numpy(sparse_mx.data)
        shape   = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, build_item_graph=False):
        # 모달별 projected feature
        image_feats = None
        text_feats  = None
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats  = self.text_trs(self.text_embedding.weight)

        if build_item_graph:
            weight = self.softmax(self.modal_weight)

            learned_adj = None
            original_adj = None

            if self.v_feat is not None:
                img_knn = build_knn_sparse_from_feats(image_feats, self.knn_k, self.device, row_bs=1024)
                img_knn = sparse_normalized_adj(img_knn)
                learned_adj  = img_knn
                original_adj = self.image_original_adj

            if self.t_feat is not None:
                txt_knn = build_knn_sparse_from_feats(text_feats, self.knn_k, self.device, row_bs=1024)
                txt_knn = sparse_normalized_adj(txt_knn)
                learned_adj = txt_knn if learned_adj is None else (learned_adj, txt_knn)

                if original_adj is None:
                    original_adj = self.text_original_adj
                else:
                    # 둘 다 있을 때 모달 가중합
                    # learned_adj가 tuple일 경우 결합
                    la = (learned_adj[0] * weight[0]) + (learned_adj[1] * weight[1])
                    oa = (self.image_original_adj * weight[0]) + (self.text_original_adj * weight[1])
                    learned_adj  = la.coalesce()
                    original_adj = oa.coalesce()

            # 정규화 및 convex 조합
            if isinstance(learned_adj, tuple):
                learned_adj = (learned_adj[0] + learned_adj[1]).coalesce()
            learned_adj = sparse_normalized_adj(learned_adj)
            original_adj = sparse_normalized_adj(original_adj)

            if self.item_adj is not None:
                del self.item_adj
            self.item_adj = ((1 - self.lambda_coeff) * learned_adj + self.lambda_coeff * original_adj).coalesce()
        else:
            # 그래프 고정
            self.item_adj = self.item_adj.detach()

        # item embedding propagation: supports sparse @ dense
        h = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.item_adj, h)

        if self.cf_model == 'ngcf':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings  = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings  = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings

        elif self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for _ in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings

        elif self.cf_model == 'mf':
            return self.user_embedding.weight, self.item_id_embedding.weight + F.normalize(h, p=2, dim=1)

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward(self.norm_adj, build_item_graph=self.build_item_graph)
        self.build_item_graph = False

        u_g_embeddings    = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )
        return batch_mf_loss + batch_emb_loss + batch_reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward(self.norm_adj, build_item_graph=True)
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores