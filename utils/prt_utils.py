import torch
from scene.gaussian_model import GaussianModel
from scene.transfer_mlp import TransferMLP

class PRTutils:
    
    @staticmethod
    def cal_diffuse(gaussian: GaussianModel, mask = None):
        """
        计算 PRT 漫反射颜色（view-independent 部分）。

        公式：C_d = ρ_d · ReLU(Σ(c_j · c_j^t) + 0.5)
        文档 LINK - ./prt_utils.md#cal_diffuse
        """
        # 获取漫反射颜色（反照率 ρ_d），可选 mask 筛选子集
        diffuse_tint = gaussian.get_diffuse_tint.contiguous() if mask is None else gaussian.get_diffuse_tint.contiguous()[mask]

        # 当前激活的 SH 阶数，确定要使用的 SH 系数数量
        deg = gaussian.active_sh_degree
        use_len = (deg + 1) ** 2

        # 取出全局 SH 光照系数 c_j（形状: [N, 3, (deg+1)^2]），截取到当前阶数
        shs = gaussian.get_shs if mask is None else gaussian.get_shs[mask]
        shs_direct_light = shs.transpose(1, 2)[..., :use_len]

        # 取出每个 Gaussian 的 SH 漫反射传输系数 c_j^t，同样截取到当前阶数
        shs_transfer = gaussian.get_diffuse_transfer if mask is None else  gaussian.get_diffuse_transfer[mask]
        shs_diffust_transfer = shs_transfer.transpose(1, 2)[..., :use_len]

        # 传输函数 T = ReLU(Σ(c_j · c_j^t) + 0.5)：逐 SH 分量点乘后求和，加偏置再 ReLU
        transport = torch.relu((shs_diffust_transfer * shs_direct_light).sum(-1) + 0.5)

        # 最终漫反射颜色 = 反照率 × 传输函数（逐 Gaussian/逐像素）
        cd = (diffuse_tint) * transport

        prt_color = cd

        return prt_color



    @staticmethod
    def cal_specular(gaussian: GaussianModel, net: TransferMLP, dir, normal, mask = None):
        """
        计算 PRT specular（view-dependent 部分，含神经辐射传输）。

        与漫反射不同，高光需要：
        1. 基于反射方向（而非固定传输）来评估传输系数
        2. 反射方向 r = 2*(n·v)*n - v，依赖视角
        3. 传输系数由 MLP 网络 G(f_t, r) 生成（神经辐射传输）
        4. 最后与 SH 光照做点积：C_s = ρ_s · ReLU(Σ(LT · c_j))
        """
        # 法线
        normal_use = normal if mask is None else normal[mask]

        # 当前 SH 阶数，截取对应数量的系数
        deg = gaussian.active_sh_degree
        use_len = (deg + 1) ** 2

        # 计算反射方向 r = 2*(n·v)*n - v
        view_dir = dir
        reflect_dir = 2.0 * (normal_use * view_dir).sum(-1, keepdims=True).clamp(min=0.0) * normal_use - view_dir

        # 通过 MLP 网络获取反射方向上的高光传输系数 LT，截取到当前 SH 阶数
        LT_coeff = PRTutils.cal_spec_coff(gaussian, net, reflect_dir, mask).unsqueeze(1)[..., :use_len]
        # 取出全局 SH 光照系数（与漫反射共用），截取到当前阶数
        shs = gaussian.get_shs if mask is None else gaussian.get_shs[mask]
        direct_light_shs = shs.transpose(1, 2).view(-1, 3, (gaussian.max_sh_degree + 1) ** 2)[..., :use_len]

        # 高光光照 = ReLU(Σ(LT · c_j))：LT 与 SH 光照逐分量点乘后求和
        direct_color = torch.relu((LT_coeff * direct_light_shs).sum(-1))
        # 高光颜色 = 高光反照率 × 高光直接光照
        specular_tint = gaussian.get_specular_tint if mask is None else gaussian.get_specular_tint[mask]
        cs = specular_tint * direct_color

        return cs

    @staticmethod
    def cal_spec_coff(gaussian: GaussianModel, net: TransferMLP, dir, mask = None):
        """
        通过神经辐射传输 MLP 计算高光 SH 传输系数。

        公式：LT = G(f_t, r)
          - f_t：每个 Gaussian 的高光特征向量（specular_feature），编码了材质属性
          - r：反射方向（view-dependent）
          - G：共享的 TransferMLP 网络
          - 输出 LT：在 SH 空间中的高光传输系数，后续与 SH 光照点积得到高光颜色
        """
        # 取高光特征（材质编码），可选 mask 筛选
        spec_feature = gaussian.get_specular_feature if mask is None else gaussian.get_specular_feature[mask]
        # 旧版实现是将特征、颜色、方向拼接后输入网络，现改为 (特征, 方向) 分离输入
        # spec_coff = net.forward(torch.cat((spec_feature, specular_tint, dir), dim=-1) )
        spec_coff = net.forward(spec_feature, dir)
        return spec_coff


    @staticmethod
    def cal_color(gaussian: GaussianModel, net: TransferMLP, dir, normal, is_training = False, mask = None):
        """
        计算完整 PRT 颜色 = 漫反射 + 高光。

        这就是 RTR-GS 前向渲染中低频率部分的最终输出，
        高频率反射成分在 deferred pass 中另外计算。
        """
        # 漫反射分量（view-independent，不依赖视角）
        diffuse_color = PRTutils.cal_diffuse(gaussian, mask)
        # 高光分量（view-dependent，依赖视角和法线方向）
        cs = PRTutils.cal_specular(gaussian, net, dir, normal, mask=mask)

        # 组合：C_prt = C_d + C_s
        prt_color = diffuse_color + cs
        return prt_color

