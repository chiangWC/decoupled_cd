import torch
import torch.nn as nn
import torch.nn.functional as F


class PosLinear(nn.Linear):
    def forward(self, input):
        weight = 2 * F.relu(torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class SVGCDNet(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n, args, pos_graph, neg_graph, device):
        super().__init__()
        self.student_n = student_n
        self.exer_n = exer_n
        self.knowledge_n = knowledge_n
        self.args = args
        self.device = device
        self.tau = args.cl_tau

        self.stu_emb = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(student_n, args.emb_dim, device=device))
        )
        self.exer_emb = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(exer_n, args.emb_dim, device=device))
        )
        self.cpt_emb = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(knowledge_n, args.emb_dim, device=device))
        )
        self.exer_disc = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(exer_n, 1, device=device))
        )

        self.prednet_stu = PosLinear(knowledge_n, knowledge_n)
        self.prednet_exer = PosLinear(knowledge_n, knowledge_n)
        self.prednet_1 = PosLinear(knowledge_n, args.dnn_units[0])
        self.drop_1 = nn.Dropout(p=args.dropout_rate)
        self.prednet_2 = PosLinear(args.dnn_units[0], args.dnn_units[1])
        self.drop_2 = nn.Dropout(p=args.dropout_rate)
        self.prednet_final = PosLinear(args.dnn_units[1], 1)

        self.se_ct_norm_graph = pos_graph.coalesce().to(device)
        self.se_ict_norm_graph = neg_graph.coalesce().to(device)

        self.ct_ind_upper, self.ct_ind_lower, self.ct_ind = self._get_upper_lower(self.se_ct_norm_graph)
        self.ict_ind_upper, self.ict_ind_lower, self.ict_ind = self._get_upper_lower(self.se_ict_norm_graph)

        self.ct_encoder_std = nn.Sequential(
            nn.Linear(knowledge_n, knowledge_n),
            nn.Linear(knowledge_n, knowledge_n),
            nn.Softplus(),
        )
        self.ict_encoder_std = nn.Sequential(
            nn.Linear(knowledge_n, knowledge_n),
            nn.Linear(knowledge_n, knowledge_n),
            nn.Softplus(),
        )
        self.ct_decoder = nn.Sequential(
            nn.Linear(knowledge_n, knowledge_n),
            nn.Linear(knowledge_n, 1),
        )
        self.ict_decoder = nn.Sequential(
            nn.Linear(knowledge_n, knowledge_n),
            nn.Linear(knowledge_n, 1),
        )

    @staticmethod
    def _get_upper_lower(graph):
        ind_row, ind_col = graph._indices()
        mask_upper = ind_row <= ind_col
        ind_upper = torch.stack((ind_row[mask_upper], ind_col[mask_upper]), dim=0)
        mask_lower = ind_row > ind_col
        ind_lower = torch.stack((ind_row[mask_lower], ind_col[mask_lower]), dim=0)
        ind_all = torch.cat((ind_upper, ind_lower), dim=1)
        return ind_upper, ind_lower, ind_all

    def _get_stu_exer_to_cpt(self):
        stu_state = self.stu_emb @ self.cpt_emb.t()
        exer_diff = self.exer_emb @ self.cpt_emb.t()
        return stu_state, exer_diff

    def graph_emb_compute(self, se_ct_adj, se_ict_adj, x_emb, y_emb, reduction="sum", dropout=False):
        all_embed = torch.cat((x_emb, y_emb), dim=0)
        emb_lists = [all_embed]
        agg_emb_ct, agg_emb_ict = all_embed, all_embed
        emb_ct_lists, emb_ict_lists = [], []

        for _ in range(self.args.n_gnn_layer):
            if not dropout:
                agg_emb_ct = torch.sparse.mm(se_ct_adj, emb_lists[-1])
                agg_emb_ict = torch.sparse.mm(se_ict_adj, emb_lists[-1])
                agg_emb = agg_emb_ct + agg_emb_ict
                emb_lists.append(agg_emb)
            else:
                agg_emb_ct = torch.sparse.mm(se_ct_adj, agg_emb_ct)
                agg_emb_ict = torch.sparse.mm(se_ict_adj, agg_emb_ict)
                emb_ct_lists.append(agg_emb_ct)
                emb_ict_lists.append(agg_emb_ict)

        if reduction == "mean":
            if not dropout:
                light_out = torch.mean(torch.stack(emb_lists, dim=1), dim=1)
                stu_abit_all, exer_diff_all = [light_out[: self.student_n]], [light_out[self.student_n :]]
            else:
                light_out_ct = torch.mean(torch.stack(emb_ct_lists, dim=1), dim=1)
                light_out_ict = torch.mean(torch.stack(emb_ict_lists, dim=1), dim=1)
                stu_abit_all = [light_out_ct[: self.student_n], light_out_ict[: self.student_n]]
                exer_diff_all = [light_out_ct[self.student_n :], light_out_ict[self.student_n :]]
        else:
            if not dropout:
                light_out = torch.sum(torch.stack(emb_lists, dim=1), dim=1)
                stu_abit_all, exer_diff_all = [light_out[: self.student_n]], [light_out[self.student_n :]]
            else:
                light_out_ct = torch.sum(torch.stack(emb_ct_lists, dim=1), dim=1)
                light_out_ict = torch.sum(torch.stack(emb_ict_lists, dim=1), dim=1)
                stu_abit_all = [light_out_ct[: self.student_n], light_out_ict[: self.student_n]]
                exer_diff_all = [light_out_ct[self.student_n :], light_out_ict[self.student_n :]]
        return stu_abit_all, exer_diff_all

    def vgae_encoder(self, stu_abit_all, exer_diff_all):
        ct_node_emb = torch.cat([stu_abit_all[0], exer_diff_all[0]], dim=0)
        ict_node_emb = torch.cat([stu_abit_all[1], exer_diff_all[1]], dim=0)

        ct_mean, ct_std = ct_node_emb, self.ct_encoder_std(ct_node_emb)
        ict_mean, ict_std = ict_node_emb, self.ict_encoder_std(ict_node_emb)
        ct_noise = torch.randn_like(ct_mean)
        ict_noise = torch.randn_like(ict_mean)

        ct_ret = ct_noise * torch.exp(ct_std) + ct_mean
        ict_ret = ict_noise * torch.exp(ict_std) + ict_mean
        return [ct_ret, ct_mean, ct_std], [ict_ret, ict_mean, ict_std]

    @staticmethod
    def _graph_norm(graph):
        graph = graph.coalesce()
        ind = graph._indices()
        row, col = ind[0], ind[1]
        rowsum = torch.sparse.sum(graph, dim=-1).to_dense()
        d_inv_sqrt = torch.pow(rowsum.clamp(min=1e-10), -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        values = graph._values() * d_inv_sqrt[row] * d_inv_sqrt[col]
        return torch.sparse_coo_tensor(ind, values, graph.shape, device=graph.device).coalesce()

    def vgae_generate(self, stu_abit_all, exer_diff_all):
        ct_info, ict_info = self.vgae_encoder(stu_abit_all, exer_diff_all)

        ct_res_input = torch.cat(
            (
                ct_info[0][self.ct_ind_upper[0]] - ct_info[0][self.ct_ind_upper[1]],
                ct_info[0][self.ct_ind_lower[1]] - ct_info[0][self.ct_ind_lower[0]],
            ),
            dim=0,
        )
        ct_edge_pred = torch.sigmoid(self.ct_decoder(ct_res_input)).view(-1)

        ict_res_input = torch.cat(
            (
                ict_info[0][self.ict_ind_upper[1]] - ict_info[0][self.ict_ind_upper[0]],
                ict_info[0][self.ict_ind_lower[0]] - ict_info[0][self.ict_ind_lower[1]],
            ),
            dim=0,
        )
        ict_edge_pred = torch.sigmoid(self.ict_decoder(ict_res_input)).view(-1)

        se_ct_graph = torch.sparse_coo_tensor(
            self.ct_ind, ct_edge_pred, self.se_ct_norm_graph.shape, device=self.device
        ).coalesce()
        se_ict_graph = torch.sparse_coo_tensor(
            self.ict_ind, ict_edge_pred, self.se_ict_norm_graph.shape, device=self.device
        ).coalesce()
        return self._graph_norm(se_ct_graph), self._graph_norm(se_ict_graph)

    def graph_representations(self, dropout=False, vgae_kl=False):
        stu_state, exer_diff = self._get_stu_exer_to_cpt()
        if dropout:
            stu_abit_all, exer_diff_all = self.graph_emb_compute(
                self.se_ct_norm_graph,
                self.se_ict_norm_graph,
                stu_state,
                exer_diff,
                reduction="mean",
                dropout=True,
            )
            if vgae_kl:
                se_ct_adj_new, se_ict_adj_new = self.vgae_generate(stu_abit_all, exer_diff_all)
                stu_abit_mix, exer_diff_mix = self.graph_emb_compute(
                    se_ct_adj_new, se_ict_adj_new, stu_state, exer_diff, reduction="sum", dropout=False
                )
                return stu_abit_all, exer_diff_all, stu_abit_mix, exer_diff_mix
            with torch.no_grad():
                se_ct_adj_new, se_ict_adj_new = self.vgae_generate(stu_abit_all, exer_diff_all)
            return self.graph_emb_compute(
                se_ct_adj_new, se_ict_adj_new, stu_state, exer_diff, reduction="sum", dropout=True
            )

        stu_abit_all, exer_diff_all = self.graph_emb_compute(
            self.se_ct_norm_graph,
            self.se_ict_norm_graph,
            stu_state,
            exer_diff,
            reduction="sum",
            dropout=False,
        )
        return stu_abit_all[0], exer_diff_all[0]

    def forward(self, batch_s_emb, batch_e_emb, batch_e_disc, e_qmat):
        batch_s_emb = torch.sigmoid(self.prednet_stu(batch_s_emb))
        batch_e_emb = torch.sigmoid(self.prednet_exer(batch_e_emb))
        batch_e_disc = torch.sigmoid(batch_e_disc)
        out = e_qmat * (batch_s_emb - batch_e_emb) * batch_e_disc
        out = self.drop_1(torch.tanh(self.prednet_1(out)))
        out = self.drop_2(torch.tanh(self.prednet_2(out)))
        return torch.sigmoid(self.prednet_final(out))

    def forward_test(self, stu_id, exer_id, Q_mat):
        stu_abit, exer_diff = self.graph_representations()
        return self(
            stu_abit[stu_id],
            exer_diff[exer_id],
            self.exer_disc[exer_id],
            Q_mat[exer_id],
        ).flatten()

    def graph_cl_learning(self, stu_abit_all, exer_diff_all, bs_stu_id, bs_exer_id):
        loss = 0.0
        selected_stus = [stu_abit_all[0][bs_stu_id], stu_abit_all[1][bs_stu_id]]
        selected_exers = [exer_diff_all[0][bs_exer_id], exer_diff_all[1][bs_exer_id]]
        for embed in [selected_stus, selected_exers]:
            if embed[0].numel() == 0 or embed[1].numel() == 0:
                continue
            embed0 = F.normalize(embed[0], p=2, dim=-1)
            embed1 = F.normalize(embed[1], p=2, dim=-1)
            ratings = torch.matmul(embed0, embed1.t())
            ratings_diag = torch.diag(ratings)
            numerator = torch.exp(ratings_diag / self.tau)
            denominator = torch.sum(torch.exp(ratings / self.tau), dim=1).clamp(min=1e-10)
            loss = loss + (-torch.mean(torch.log((numerator / denominator).clamp(min=1e-10))))
        return loss

    def cal_loss(self, stu_id, exer_id, label, Q_mat):
        stu_abit, exer_diff = self.graph_representations()
        y_pd = self(
            stu_abit[stu_id],
            exer_diff[exer_id],
            self.exer_disc[exer_id],
            Q_mat[exer_id],
        ).flatten()
        bce_loss = F.binary_cross_entropy(y_pd, label)
        return bce_loss, {"bce_loss": bce_loss}

    def cal_loss_cl(self, stu_id, exer_id, label, Q_mat):
        stu_ct_abit_all, exer_ct_diff_all = [], []
        stu_ict_abit_all, exer_ict_diff_all = [], []
        for _ in [0, 1]:
            stu_abit_all_new, exer_diff_all_new = self.graph_representations(dropout=True)
            stu_ct_abit_all.append(stu_abit_all_new[0])
            exer_ct_diff_all.append(exer_diff_all_new[0])
            stu_ict_abit_all.append(stu_abit_all_new[1])
            exer_ict_diff_all.append(exer_diff_all_new[1])

        pos_mask = label == 1
        neg_mask = label == 0
        ct_cl_loss = self.graph_cl_learning(
            stu_ct_abit_all, exer_ct_diff_all, stu_id[pos_mask], exer_id[pos_mask]
        )
        ict_cl_loss = self.graph_cl_learning(
            stu_ict_abit_all, exer_ict_diff_all, stu_id[neg_mask], exer_id[neg_mask]
        )
        pos_w = float(pos_mask.sum().item()) / max(1, len(stu_id))
        neg_w = float(neg_mask.sum().item()) / max(1, len(stu_id))
        cl_loss = (ct_cl_loss * pos_w + ict_cl_loss * neg_w) * self.args.cl_weight
        return cl_loss, {"cl_loss": cl_loss}

    def cal_loss_kl(self, stu_id, exer_id, label, Q_mat):
        stu_abit_all, exer_diff_all, stu_abit_mix, exer_diff_mix = self.graph_representations(
            dropout=True, vgae_kl=True
        )
        ct_info, ict_info = self.vgae_encoder(stu_abit_all, exer_diff_all)

        ct_kl = -0.5 * (1 + 2 * torch.log(ct_info[2].clamp(min=1e-10)) - ct_info[1] ** 2 - ct_info[2] ** 2).sum(dim=1)
        ict_kl = -0.5 * (1 + 2 * torch.log(ict_info[2].clamp(min=1e-10)) - ict_info[1] ** 2 - ict_info[2] ** 2).sum(dim=1)
        pos_mask = label == 1
        neg_mask = label == 0
        pos_w = float(pos_mask.sum().item()) / max(1, len(stu_id))
        neg_w = float(neg_mask.sum().item()) / max(1, len(stu_id))
        kl_div = (ct_kl * pos_w + ict_kl * neg_w) * self.args.beta

        y_pd = self(
            stu_abit_mix[0][stu_id],
            exer_diff_mix[0][exer_id],
            self.exer_disc[exer_id],
            Q_mat[exer_id],
        ).flatten()
        res_bce = F.binary_cross_entropy(y_pd, label, reduction="none")
        loss = (res_bce + kl_div.mean()).mean()
        return loss, {"res_bce_loss": res_bce.mean(), "kl_loss": kl_div.mean()}
