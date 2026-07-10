from collections import OrderedDict

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def none_neg_clipper(module):
    if hasattr(module, "weight") and module.weight is not None:
        module.weight.data.clamp_(min=0.0)


def dot_product(a, b):
    return torch.sigmoid(torch.matmul(a, b.t()))


def polynomial_kernel_dot_product(a, b, constant=0.1, degree=2.0):
    return torch.sigmoid((torch.matmul(a, b.t()) + constant) ** degree)


def rbf_kernel_dot_product(a, b, sigma=0.1):
    dist = torch.norm(a[:, None] - b, dim=2)
    return torch.sigmoid(torch.exp(-(dist**2) / (2 * sigma**2)))


class DP_IF(nn.Module):
    def __init__(self, knowledge_num, hidden_dims, dropout, device, dtype, kernel):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
        self.kernel = kernel
        self.transform_kernel = self.get_kernel()

        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers["linear0"] = nn.Linear(self.knowledge_num, hidden_dim, dtype=self.dtype)
                layers["activation0"] = nn.Tanh()
            else:
                layers[f"dropout{idx}"] = nn.Dropout(p=self.dropout)
                layers[f"linear{idx}"] = nn.Linear(self.hidden_dims[idx - 1], hidden_dim, dtype=self.dtype)
                layers[f"activation{idx}"] = nn.Tanh()
        last_idx = len(self.hidden_dims)
        layers[f"dropout{last_idx}"] = nn.Dropout(p=self.dropout)
        layers[f"linear{last_idx}"] = nn.Linear(self.hidden_dims[-1], 1, dtype=self.dtype)
        layers[f"activation{last_idx}"] = nn.Sigmoid()
        self.mlp = nn.Sequential(layers).to(self.device)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def get_kernel(self):
        if self.kernel == "dp-linear":
            return dot_product
        if self.kernel == "dp-poly":
            return polynomial_kernel_dot_product
        if self.kernel == "dp-rbf":
            return rbf_kernel_dot_product
        raise ValueError(f"Unsupported kernel: {self.kernel}")

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        knowledge_ts = kwargs["knowledge_ts"]
        q_mask = kwargs["q_mask"]
        input_x = torch.sigmoid(disc_ts) * (
            self.transform_kernel(student_ts, knowledge_ts) - self.transform_kernel(diff_ts, knowledge_ts)
        ) * q_mask
        return self.mlp(input_x).view(-1)

    def transform(self, mastery, knowledge):
        return torch.sigmoid(self.transform_kernel(mastery, knowledge))

    def monotonicity(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)


class NCD_IF(nn.Module):
    def __init__(self, knowledge_num, hidden_dims, dropout, device, dtype):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers["linear0"] = nn.Linear(self.knowledge_num, hidden_dim, dtype=self.dtype)
                layers["activation0"] = nn.Tanh()
            else:
                layers[f"dropout{idx}"] = nn.Dropout(p=self.dropout)
                layers[f"linear{idx}"] = nn.Linear(self.hidden_dims[idx - 1], hidden_dim, dtype=self.dtype)
                layers[f"activation{idx}"] = nn.Tanh()
        last_idx = len(self.hidden_dims)
        layers[f"dropout{last_idx}"] = nn.Dropout(p=self.dropout)
        layers[f"linear{last_idx}"] = nn.Linear(self.hidden_dims[-1], 1, dtype=self.dtype)
        layers[f"activation{last_idx}"] = nn.Sigmoid()
        self.mlp = nn.Sequential(layers).to(self.device)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        q_mask = kwargs["q_mask"]
        input_x = torch.sigmoid(disc_ts) * (torch.sigmoid(student_ts) - torch.sigmoid(diff_ts)) * q_mask
        return self.mlp(input_x).view(-1)

    def transform(self, mastery, knowledge):
        return torch.sigmoid(mastery)

    def monotonicity(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)


class IRT_IF(nn.Module):
    def __init__(self, device, dtype, latent_dim=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.latent_dim = latent_dim
        if self.latent_dim is not None:
            self.transform_student = nn.Linear(latent_dim, 1, dtype=dtype).to(self.device)
            self.transform_exercise = nn.Linear(latent_dim, 1, dtype=dtype).to(self.device)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        if self.latent_dim is not None:
            input_x = torch.sigmoid(self.transform_student(student_ts) - self.transform_exercise(diff_ts))
        else:
            input_x = torch.sigmoid(student_ts - diff_ts)
        return input_x.view(-1)

    def transform(self, mastery, knowledge):
        return torch.sigmoid(mastery)

    def monotonicity(self):
        return None


class MIRT_IF(nn.Module):
    def __init__(self, knowledge_num, latent_dim, device, dtype, utlize=False):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.device = device
        self.dtype = dtype
        self.ultize = utlize

    @staticmethod
    def irt2pl(theta, a, b, F_module=torch):
        return (1 / (1 + F_module.exp(-F_module.sum(F_module.multiply(a, theta), dim=-1) + b))).view(-1)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        student_ts = torch.squeeze(student_ts, dim=-1)
        diff_ts = torch.squeeze(diff_ts, dim=-1)
        disc_ts = torch.squeeze(disc_ts, dim=-1)
        return self.irt2pl(student_ts, F.softplus(diff_ts), disc_ts)

    def transform(self, mastery, knowledge):
        return torch.sigmoid(mastery)

    def monotonicity(self):
        return None


class KANCD_IF(nn.Module):
    def __init__(self, knowledge_num, latent_dim, hidden_dims, dropout, device, dtype):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        self.k_diff_full = nn.Linear(self.latent_dim, 1, dtype=dtype).to(self.device)
        self.stat_full = nn.Linear(self.latent_dim, 1, dtype=dtype).to(self.device)

        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers["linear0"] = nn.Linear(self.knowledge_num, hidden_dim, dtype=self.dtype)
                layers["activation0"] = nn.Tanh()
            else:
                layers[f"dropout{idx}"] = nn.Dropout(p=self.dropout)
                layers[f"linear{idx}"] = nn.Linear(self.hidden_dims[idx - 1], hidden_dim, dtype=self.dtype)
                layers[f"activation{idx}"] = nn.Tanh()
        last_idx = len(self.hidden_dims)
        layers[f"dropout{last_idx}"] = nn.Dropout(p=self.dropout)
        layers[f"linear{last_idx}"] = nn.Linear(self.hidden_dims[-1], 1, dtype=self.dtype)
        layers[f"activation{last_idx}"] = nn.Sigmoid()
        self.mlp = nn.Sequential(layers).to(self.device)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        knowledge_ts = kwargs["knowledge_ts"]
        q_mask = kwargs["q_mask"]

        batch, dim = student_ts.size()
        stu_emb = student_ts.view(batch, 1, dim).repeat(1, self.knowledge_num, 1)
        knowledge_emb = knowledge_ts.repeat(batch, 1).view(batch, self.knowledge_num, -1)
        exer_emb = diff_ts.view(batch, 1, dim).repeat(1, self.knowledge_num, 1)
        input_x = torch.sigmoid(disc_ts) * (
            torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
            - torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        ) * q_mask
        return self.mlp(input_x).view(-1)

    def transform(self, mastery, knowledge):
        self.eval()
        blocks = torch.split(torch.arange(mastery.shape[0], device=self.device), 5)
        mas = []
        for block in blocks:
            batch, dim = mastery[block].size()
            stu_emb = mastery[block].view(batch, 1, dim).repeat(1, self.knowledge_num, 1)
            knowledge_emb = knowledge.repeat(batch, 1).view(batch, self.knowledge_num, -1)
            mas.append(torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1))
        return torch.vstack(mas)

    def monotonicity(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)


class CDMFKC_IF(nn.Module):
    def __init__(self, g_impact_a, g_impact_b, knowledge_num, hidden_dims, dropout, device, dtype, latent_dim=None):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.g_impact_a = g_impact_a
        self.g_impact_b = g_impact_b
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
        self.latent_dim = latent_dim
        if latent_dim is not None:
            self.transform_impact = nn.Linear(latent_dim, knowledge_num, dtype=dtype).to(self.device)

        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers["linear0"] = nn.Linear(self.knowledge_num, hidden_dim, dtype=self.dtype)
                layers["activation0"] = nn.Tanh()
            else:
                layers[f"dropout{idx}"] = nn.Dropout(p=self.dropout)
                layers[f"linear{idx}"] = nn.Linear(self.hidden_dims[idx - 1], hidden_dim, dtype=self.dtype)
                layers[f"activation{idx}"] = nn.Tanh()
        last_idx = len(self.hidden_dims)
        layers[f"dropout{last_idx}"] = nn.Dropout(p=self.dropout)
        layers[f"linear{last_idx}"] = nn.Linear(self.hidden_dims[-1], 1, dtype=self.dtype)
        layers[f"activation{last_idx}"] = nn.Sigmoid()
        self.mlp = nn.Sequential(layers).to(self.device)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        q_mask = kwargs["q_mask"]
        other = kwargs.get("other", {})
        if self.latent_dim is not None:
            h_impact = self.transform_impact(torch.sigmoid(other["knowledge_impact"]))
        else:
            h_impact = torch.sigmoid(other["knowledge_impact"])
        g_impact = torch.sigmoid(
            self.g_impact_a * h_impact + self.g_impact_b * torch.sigmoid(diff_ts) * torch.sigmoid(disc_ts)
        )
        input_x = torch.sigmoid(disc_ts) * (torch.sigmoid(student_ts) + g_impact - torch.sigmoid(diff_ts)) * q_mask
        return self.mlp(input_x).view(-1)

    def transform(self, mastery, knowledge):
        return torch.sigmoid(mastery)

    def monotonicity(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)


class KSCD_IF(nn.Module):
    def __init__(self, dropout, knowledge_num, latent_dim, device, dtype):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.device = device
        self.dtype = dtype

        self.prednet_full1 = nn.Linear(
            self.knowledge_num + self.latent_dim, self.knowledge_num, bias=False, dtype=dtype
        ).to(self.device)
        self.drop_1 = nn.Dropout(p=dropout)
        self.prednet_full2 = nn.Linear(
            self.knowledge_num + self.latent_dim, self.knowledge_num, bias=False, dtype=dtype
        ).to(self.device)
        self.drop_2 = nn.Dropout(p=dropout)
        self.prednet_full3 = nn.Linear(self.knowledge_num, 1, dtype=dtype).to(self.device)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        q_mask = kwargs["q_mask"]
        knowledge_ts = kwargs["knowledge_ts"]

        stu_ability = torch.mm(student_ts, knowledge_ts.t()).sigmoid()
        exer_diff = torch.mm(diff_ts, knowledge_ts.t()).sigmoid()
        batch_stu_vector = stu_ability.repeat(1, self.knowledge_num).reshape(
            stu_ability.shape[0], self.knowledge_num, stu_ability.shape[1]
        )
        batch_exer_vector = exer_diff.repeat(1, self.knowledge_num).reshape(
            exer_diff.shape[0], self.knowledge_num, exer_diff.shape[1]
        )
        kn_vector = knowledge_ts.repeat(stu_ability.shape[0], 1).reshape(
            stu_ability.shape[0], self.knowledge_num, self.latent_dim
        )

        preference = torch.tanh(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.tanh(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * q_mask.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(q_mask, dim=1).unsqueeze(1)
        y_pd = sum_out / count_of_concept
        return y_pd.view(-1)

    def transform(self, mastery, knowledge):
        stu_mastery = torch.mm(mastery, knowledge.t()).sigmoid()
        stu_vector = stu_mastery.repeat(1, self.knowledge_num).reshape(
            stu_mastery.shape[0], self.knowledge_num, stu_mastery.shape[1]
        )
        kn_vector = knowledge.repeat(stu_mastery.shape[0], 1).reshape(
            stu_mastery.shape[0], self.knowledge_num, self.latent_dim
        )
        preference = torch.tanh(self.prednet_full1(torch.cat((stu_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference))
        return o.squeeze(-1)

    def monotonicity(self):
        self.prednet_full1.apply(none_neg_clipper)
        self.prednet_full2.apply(none_neg_clipper)
        self.prednet_full3.apply(none_neg_clipper)


class ORCDFExtractor(nn.Module):
    def __init__(
        self,
        student_num,
        exercise_num,
        knowledge_num,
        latent_dim=32,
        device="cpu",
        dtype=torch.float64,
        gcn_layers=3,
        keep_prob=0.9,
        mode="all",
        ssl_temp=0.8,
        ssl_weight=1e-2,
    ):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.device = device
        self.dtype = dtype
        self.gcn_layers = gcn_layers
        self.keep_prob = keep_prob
        self.mode = mode
        self.ssl_temp = ssl_temp
        self.ssl_weight = ssl_weight
        self.gcn_drop = True
        self.graph_dict = None

        self._student_emb = nn.Embedding(student_num, latent_dim, dtype=dtype).to(device)
        self._exercise_emb = nn.Embedding(exercise_num, latent_dim, dtype=dtype).to(device)
        self._knowledge_emb = nn.Embedding(knowledge_num, latent_dim, dtype=dtype).to(device)
        self._disc_emb = nn.Embedding(exercise_num, 1, dtype=dtype).to(device)
        self._knowledge_impact_emb = nn.Embedding(exercise_num, latent_dim, dtype=dtype).to(device)

        self.concat_layer = nn.Linear(2 * latent_dim, latent_dim, dtype=dtype).to(device)
        self.transfer_student_layer = nn.Linear(latent_dim, knowledge_num, dtype=dtype).to(device)
        self.transfer_exercise_layer = nn.Linear(latent_dim, knowledge_num, dtype=dtype).to(device)
        self.transfer_knowledge_layer = nn.Linear(latent_dim, knowledge_num, dtype=dtype).to(device)

        self._emb_map = {
            "mastery": self._student_emb.weight,
            "diff": self._exercise_emb.weight,
            "disc": self._disc_emb.weight,
            "knowledge": self._knowledge_emb.weight,
        }
        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def get_graph_dict(self, graph_dict):
        self.graph_dict = graph_dict

    def get_all_emb(self):
        return torch.cat(
            [self._student_emb.weight, self._exercise_emb.weight, self._knowledge_emb.weight], dim=0
        ).to(self.device)

    def _dropout_graph(self, graph):
        if not self.gcn_drop or not self.training:
            return graph

        graph = graph.coalesce()
        values = graph.values()
        indices = graph.indices()
        keep_mask = (torch.rand(values.size(0), device=values.device) + self.keep_prob).floor().bool()
        if keep_mask.sum() == 0:
            return graph
        new_indices = indices[:, keep_mask]
        new_values = values[keep_mask] / self.keep_prob
        return torch.sparse_coo_tensor(new_indices, new_values, graph.size(), dtype=graph.dtype).coalesce()

    def convolution(self, graph):
        all_emb = self.get_all_emb()
        emb = [all_emb]
        for _ in range(self.gcn_layers):
            all_emb = torch.sparse.mm(self._dropout_graph(graph), all_emb)
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return out_emb

    def _common_forward(self, right, wrong):
        all_emb = self.get_all_emb()
        emb = [all_emb]
        right_emb = all_emb
        wrong_emb = all_emb
        for _ in range(self.gcn_layers):
            right_emb = torch.sparse.mm(self._dropout_graph(right), right_emb)
            wrong_emb = torch.sparse.mm(self._dropout_graph(wrong), wrong_emb)
            all_emb = self.concat_layer(torch.cat([right_emb, wrong_emb], dim=1))
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        stu = out_emb[: self.student_num]
        exer = out_emb[self.student_num : self.student_num + self.exercise_num]
        know = out_emb[self.student_num + self.exercise_num :]
        return stu, exer, know

    @staticmethod
    def _info_nce(view1, view2, temperature=1.0, normalize=False):
        if normalize:
            view1 = F.normalize(view1, dim=1)
            view2 = F.normalize(view2, dim=1)
        logits = (view1 @ view2.t()) / temperature
        score = torch.diag(F.log_softmax(logits, dim=1))
        return -score.mean()

    def extract(self, student_id, exercise_id, q_mask):
        if "dis" not in self.mode:
            stu_forward, exer_forward, know_forward = self._common_forward(
                self.graph_dict["right"], self.graph_dict["wrong"]
            )
            stu_forward_flip, exer_forward_flip, _ = self._common_forward(
                self.graph_dict["right_flip"], self.graph_dict["wrong_flip"]
            )
        else:
            out = self.convolution(self.graph_dict["all"])
            stu_forward = out[: self.student_num]
            exer_forward = out[self.student_num : self.student_num + self.exercise_num]
            know_forward = out[self.student_num + self.exercise_num :]
            stu_forward_flip = stu_forward
            exer_forward_flip = exer_forward

        extra_loss = self._student_emb.weight.new_tensor(0.0)
        if "cl" not in self.mode:
            extra_loss = self.ssl_weight * (
                self._info_nce(stu_forward, stu_forward_flip, temperature=self.ssl_temp)
                + self._info_nce(exer_forward, exer_forward_flip, temperature=self.ssl_temp)
            )

        if "tf" not in self.mode:
            student_ts = self.transfer_student_layer(F.embedding(student_id, stu_forward))
            diff_ts = self.transfer_exercise_layer(F.embedding(exercise_id, exer_forward))
            knowledge_ts = self.transfer_knowledge_layer(know_forward)
        else:
            student_ts = F.embedding(student_id, stu_forward)
            diff_ts = F.embedding(exercise_id, exer_forward)
            knowledge_ts = know_forward

        disc_ts = self._disc_emb(exercise_id)
        knowledge_impact = self._knowledge_impact_emb(exercise_id)
        extras = {"extra_loss": extra_loss, "knowledge_impact": knowledge_impact}
        return student_ts, diff_ts, disc_ts, knowledge_ts, extras

    def get_flip_graph(self):
        if not self.graph_dict or not self.graph_dict.get("flip_ratio"):
            return

        np_response_flip = self.graph_dict["response"].copy()
        column = np_response_flip[:, 2]
        probability = np.random.choice(
            [True, False],
            size=column.shape,
            p=[self.graph_dict["flip_ratio"], 1 - self.graph_dict["flip_ratio"]],
        )
        column[probability] = 1 - column[probability]
        np_response_flip[:, 2] = column

        se_graph_right_flip, se_graph_wrong_flip = [
            self._create_adj_se(np_response_flip, is_subgraph=True)[i] for i in range(2)
        ]
        ek_graph = self.graph_dict["Q_Matrix"]
        self.graph_dict["right_flip"] = self._final_graph(se_graph_right_flip, ek_graph)
        self.graph_dict["wrong_flip"] = self._final_graph(se_graph_wrong_flip, ek_graph)

    def __getitem__(self, item):
        if item not in self._emb_map:
            raise ValueError(f"We can only detach {self._emb_map.keys()} from embeddings.")

        if "dis" not in self.mode:
            stu_forward, exer_forward, know_forward = self._common_forward(
                self.graph_dict["right"], self.graph_dict["wrong"]
            )
        else:
            out = self.convolution(self.graph_dict["all"])
            stu_forward = out[: self.student_num]
            exer_forward = out[self.student_num : self.student_num + self.exercise_num]
            know_forward = out[self.student_num + self.exercise_num :]

        student_ts = self.transfer_student_layer(stu_forward) if "tf" not in self.mode else stu_forward
        diff_ts = self.transfer_exercise_layer(exer_forward) if "tf" not in self.mode else exer_forward
        knowledge_ts = self.transfer_knowledge_layer(know_forward) if "tf" not in self.mode else know_forward
        disc_ts = self._disc_emb.weight

        self._emb_map["mastery"] = student_ts
        self._emb_map["diff"] = diff_ts
        self._emb_map["disc"] = disc_ts
        self._emb_map["knowledge"] = knowledge_ts
        return self._emb_map[item]

    @staticmethod
    def _get_csr(rows, cols, shape):
        values = np.ones_like(rows, dtype=np.float64)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)

    @staticmethod
    def _sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

    def _create_adj_se(self, np_response, is_subgraph=False):
        if is_subgraph:
            if self.mode == "R":
                empty = np.zeros(shape=(self.student_num, self.exercise_num))
                return empty, empty

            train_stu_right = np_response[np_response[:, 2] == 1, 0]
            train_exer_right = np_response[np_response[:, 2] == 1, 1]
            train_stu_wrong = np_response[np_response[:, 2] == 0, 0]
            train_exer_wrong = np_response[np_response[:, 2] == 0, 1]

            adj_se_right = self._get_csr(
                train_stu_right, train_exer_right, shape=(self.student_num, self.exercise_num)
            )
            adj_se_wrong = self._get_csr(
                train_stu_wrong, train_exer_wrong, shape=(self.student_num, self.exercise_num)
            )
            return adj_se_right.toarray(), adj_se_wrong.toarray()

        if self.mode == "R":
            return np.zeros(shape=(self.student_num, self.exercise_num))

        response_stu = np_response[:, 0]
        response_exer = np_response[:, 1]
        adj_se = self._get_csr(response_stu, response_exer, shape=(self.student_num, self.exercise_num))
        return adj_se.toarray()

    def _final_graph(self, se, ek):
        sek_num = self.student_num + self.exercise_num + self.knowledge_num
        se_num = self.student_num + self.exercise_num
        tmp = np.zeros(shape=(sek_num, sek_num), dtype=np.float64)
        tmp[: self.student_num, self.student_num : se_num] = se
        tmp[self.student_num : se_num, se_num:sek_num] = ek
        graph = tmp + tmp.T + np.identity(sek_num, dtype=np.float64)
        graph = sp.csr_matrix(graph)

        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return self._sp_mat_to_sp_tensor(adj_matrix).to(self.device)


class ORCDFNet(nn.Module):
    def __init__(
        self,
        student_n,
        exer_n,
        knowledge_n,
        latent_dim=32,
        gcn_layers=3,
        keep_prob=0.9,
        if_type="dp-linear",
        mode="all",
        flip_ratio=0.1,
        ssl_temp=0.8,
        ssl_weight=1e-2,
        prednet_len1=512,
        prednet_len2=256,
        dropout=0.0,
        device="cpu",
        dtype=torch.float64,
    ):
        super().__init__()
        if if_type in {"kancd", "kscd", "mirt", "irt"} and not mode.startswith("tf"):
            mode = "tf" + mode
        self.mode = mode
        self.flip_ratio = flip_ratio
        self.extractor = ORCDFExtractor(
            student_num=student_n,
            exercise_num=exer_n,
            knowledge_num=knowledge_n,
            latent_dim=latent_dim,
            device=device,
            dtype=dtype,
            gcn_layers=gcn_layers,
            keep_prob=keep_prob,
            mode=mode,
            ssl_temp=ssl_temp,
            ssl_weight=ssl_weight,
        )
        hidden_dims = [prednet_len1, prednet_len2]
        if if_type == "ncd":
            self.inter_func = NCD_IF(
                knowledge_num=knowledge_n,
                hidden_dims=hidden_dims,
                dropout=0.0,
                device=device,
                dtype=dtype,
            )
        elif if_type.startswith("dp"):
            self.inter_func = DP_IF(
                knowledge_num=knowledge_n,
                hidden_dims=hidden_dims,
                dropout=0.0,
                device=device,
                dtype=dtype,
                kernel=if_type,
            )
        elif if_type == "mirt":
            self.inter_func = MIRT_IF(
                knowledge_num=knowledge_n,
                latent_dim=latent_dim,
                device=device,
                dtype=dtype,
                utlize=True,
            )
        elif if_type == "kancd":
            self.inter_func = KANCD_IF(
                knowledge_num=knowledge_n,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=0.5,
                device=device,
                dtype=dtype,
            )
        elif if_type == "cdmfkc":
            self.inter_func = CDMFKC_IF(
                g_impact_a=0.5,
                g_impact_b=0.5,
                knowledge_num=knowledge_n,
                hidden_dims=hidden_dims,
                dropout=0.5,
                device=device,
                dtype=dtype,
                latent_dim=latent_dim,
            )
        elif if_type == "irt":
            self.inter_func = IRT_IF(device=device, dtype=dtype, latent_dim=latent_dim)
        elif if_type == "kscd":
            self.inter_func = KSCD_IF(
                dropout=0.5,
                knowledge_num=knowledge_n,
                latent_dim=latent_dim,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unsupported if_type: {if_type}")

    def get_graph_dict(self, graph_dict):
        self.extractor.get_graph_dict(graph_dict)

    def get_flip_graph(self):
        self.extractor.get_flip_graph()

    def forward(self, stu_id, exer_id, kn_emb):
        student_ts, diff_ts, disc_ts, knowledge_ts, extras = self.extractor.extract(stu_id, exer_id, kn_emb)
        pred = self.inter_func.compute(
            student_ts=student_ts,
            diff_ts=diff_ts,
            disc_ts=disc_ts,
            q_mask=kn_emb,
            knowledge_ts=knowledge_ts,
            other=extras,
        )
        return pred, extras["extra_loss"]
