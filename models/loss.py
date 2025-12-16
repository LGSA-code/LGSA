import torch
# new
import torch.nn as nn
import torch.nn.functional as F
import math

def multilabel_categorical_crossentropy(y_true, y_pred):
    loss_mask = y_true != -100
    y_true = y_true.masked_select(loss_mask).view(-1, y_pred.size(-1)) 
    y_pred = y_pred.masked_select(loss_mask).view(-1, y_true.size(-1)) 
    y_pred = (1 - 2 * y_true) * y_pred 
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[:, :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1) 
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1) 
    return (neg_loss + pos_loss).mean() 

def global_singular_smoothing_loss(label_weight):
    loss = -torch.norm(label_weight, p='nuc') / torch.sum(torch.norm(label_weight, dim=1))
    return loss

def local_singular_smoothing_loss(label_weight, depth2label, num_class):
    hie_loss = []
    loss = 0
    for depth in depth2label:
        if len(depth2label[depth]) > 1 :
            hie_loss.append((depth + 1) * (len(depth2label[depth]) / num_class) * (-torch.norm(label_weight[depth2label[depth]], p='nuc') / torch.sum(torch.norm(label_weight[depth2label[depth]], dim=1))))
    for l in hie_loss:
        loss += l
    return loss

class HierarchicalGofLoss(nn.Module):
    def __init__(self, depth2label, value2slot,
                 w_global_orth=1.0, w_parent_child=1.0,
                 w_brother_res=1.0, w_len_decay=1.0,
                 parent_child_angle_deg=15.0, length_decay_factor=0.9,
                 eps=1e-8, return_components=False):
        super().__init__()
        
        # --- 1. 预计算和准备索引 ---
        # 预计算可以向量化的父子关系索引
        child_indices, parent_indices = [], []
        for child, parent in value2slot.items():
            if parent != -1:
                child_indices.append(child)
                parent_indices.append(parent)

        # 将索引列表注册为模型的缓冲区(buffer)，这样它们会自动移动到正确的设备(如GPU)
        self.register_buffer('child_indices', torch.tensor(child_indices, dtype=torch.long))
        self.register_buffer('parent_indices', torch.tensor(parent_indices, dtype=torch.long))

        # 预计算兄弟关系，为半向量化做准备
        self.parent_to_children = {}
        for child, parent in value2slot.items():
            if parent != -1:
                if parent not in self.parent_to_children:
                    self.parent_to_children[parent] = []
                self.parent_to_children[parent].append(child)

        self.top_level_labels = depth2label.get(0, [])

        # --- 2. 保存超参数 ---
        self.w_global_orth = w_global_orth
        self.w_parent_child = w_parent_child
        self.w_brother_res = w_brother_res
        self.w_len_decay = w_len_decay
        self.target_cos = math.cos(math.radians(parent_child_angle_deg))
        self.length_decay_factor = length_decay_factor

        self.eps = eps
        self.return_components = return_components

    def forward(self, current_embeddings):
        comps = {}
        total_loss = torch.tensor(0.0, device=current_embeddings.device)

        # --- 全局正交性损失 (已向量化) ---
        if len(self.top_level_labels) > 1:
            top_vectors = current_embeddings[self.top_level_labels]
            # 通过矩阵乘法计算所有对的余弦相似度
            top_vectors_norm = F.normalize(top_vectors, p=2, dim=1)
            sim_matrix = torch.matmul(top_vectors_norm, top_vectors_norm.t())
            # 取上三角部分（不含对角线），计算损失
            # 2.0 / (N*(N-1)) 是为了取平均
            num_pairs = len(self.top_level_labels) * (len(self.top_level_labels) - 1)
            if num_pairs > 0:
                tri = torch.triu(sim_matrix, diagonal=1)
                global_orth_loss = (tri ** 2).sum() * 2.0 / num_pairs
                comps['global_orth'] = global_orth_loss
                # global_orth_loss = torch.abs(torch.triu(sim_matrix, diagonal=1)).sum() * 2.0 / num_pairs
                total_loss += self.w_global_orth * global_orth_loss

        # --- 父子相似性 & 长度衰减损失 (已完全向量化) ---
        if len(self.child_indices) > 0:
            E_children = current_embeddings[self.child_indices]
            E_parents = current_embeddings[self.parent_indices]

            # 父子相似性损失
            cos_sims = F.cosine_similarity(E_children, E_parents, dim=1, eps=self.eps)
            parent_child_loss = torch.mean((cos_sims - self.target_cos) ** 2)
            comps['parent_child'] = parent_child_loss
            total_loss += self.w_parent_child * parent_child_loss

            # 长度衰减损失
            norms_c = torch.linalg.norm(E_children, dim=1) + self.eps
            norms_p = torch.linalg.norm(E_parents, dim=1) + self.eps
            len_decay_loss = torch.mean(F.relu(self.length_decay_factor * norms_p - norms_c))
            comps['len_decay'] = len_decay_loss
            total_loss += self.w_len_decay * len_decay_loss
            
        # --- 兄弟残差正交性损失 (半向量化，显著提速) ---
        # 这部分完全向量化很困难，但我们可以向量化每个“家族内部”的计算
        brother_res_loss = torch.tensor(0.0, device=current_embeddings.device)
        num_families_with_brothers = 0
        for parent, children in self.parent_to_children.items():
            if len(children) > 1:
                num_families_with_brothers += 1
                E_p = current_embeddings[parent]
                E_children_family = current_embeddings[children]
                
                # 向量化投影计算
                dot_prods = torch.mv(E_children_family, E_p) # (K,)
                p_norm_sq = torch.dot(E_p, E_p)
                if p_norm_sq < 1e-8: continue
                
                projections = torch.outer(dot_prods / p_norm_sq, E_p) # (K, D)
                
                # 向量化残差计算
                residuals = E_children_family - projections
                
                # 向量化计算所有兄弟对的残差余弦相似度
                residuals_norm = F.normalize(residuals, p=2, dim=1)
                res_sim_matrix = torch.matmul(residuals_norm, residuals_norm.t())
                
                # 取上三角部分的均值作为这个家族的损失
                num_brother_pairs = len(children) * (len(children) - 1)
                if num_brother_pairs > 0:
                   family_loss = torch.abs(torch.triu(res_sim_matrix, diagonal=1)).sum() * 2.0 / num_brother_pairs
                   comps['brother_res'] = family_loss
                   brother_res_loss += family_loss

        if num_families_with_brothers > 0:
            total_loss += self.w_brother_res * (brother_res_loss / num_families_with_brothers)
            
        return (total_loss, comps) if self.return_components else total_loss