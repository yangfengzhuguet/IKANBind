########################################################################################################################
#
#This program is used to replace B-spline functions in KAN with functions that have explicit mathematical expressions.
########################################################################################################################


import torch
import torch.nn.functional as F
import math

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            num_basis=10,  # 每种基函数（tanh, sin, cos, x, x^2）各2个尺度
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_basis = num_basis
        self.num_basis_per_type = num_basis // 5  # 每种基函数的尺度数
        assert num_basis % 5 == 0, "num_basis must be divisible by 5 for tanh, sin, cos, x, and x^2"
        self.register_buffer("grid", None)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, num_basis)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            init_size = 100  # 用于初始化的样本数
            x = torch.linspace(-1, 1, init_size).unsqueeze(1).repeat(1, self.in_features).to(self.base_weight.device)
            noise = (
                    (torch.rand(init_size, self.in_features, self.out_features) - 0.5)
                    * self.scale_noise
            )
            print(f"reset_parameters: x shape: {x.shape}, noise shape: {noise.shape}, in_features: {self.in_features}")
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(x, noise)
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        x = x.unsqueeze(-1)
        scales = torch.arange(1, self.num_basis_per_type + 1, device=x.device).float().view(1, 1, -1)

        # 计算 tanh, sin, cos 基函数
        tanh_bases = torch.tanh(scales * math.pi * x)
        sin_bases = torch.sin(scales * math.pi * x)
        cos_bases = torch.cos(scales * math.pi * x)
        # 计算 x 和 x^ oa 基函数
        x_bases = x.repeat(1, 1, self.num_basis_per_type) * scales
        x2_bases = (x ** 2).repeat(1, 1, self.num_basis_per_type) * scales

        # 拼接五种基函数
        bases = torch.cat([tanh_bases, sin_bases, cos_bases, x_bases, x2_bases], dim=-1)
        assert bases.size() == (x.size(0), self.in_features, self.num_basis)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        print(f"curve2coeff: x shape: {x.shape}, y shape: {y.shape}, in_features: {self.in_features}")
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (self.out_features, self.in_features, self.num_basis)
        return result.contiguous()
    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    def update_grid(self, x: torch.Tensor, margin=0.01):
        pass

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / (regularization_loss_activation + 1e-8)
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


