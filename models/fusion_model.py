import torch
import torch.nn as nn


# Common fusion module
class fusion_module_C(nn.Module):
    def __init__(self, in_channels_img, in_channels_nonimg, out_channels=512):
        super(fusion_module_C, self).__init__()
        print(
            "Fusion Module C: split sigmoid weight gated point, image fusion")
        self.gate_mri = nn.Sequential(
            nn.Linear(in_channels_img, out_channels),
            nn.Sigmoid(),
        )
        self.gate_pet = nn.Sequential(
            nn.Linear(in_channels_img, out_channels),
            nn.Sigmoid(),
        )
        self.gate_demo = nn.Sequential(
            nn.Linear(in_channels_nonimg, out_channels),
            nn.Sigmoid(),
        )
        self.input_mri = nn.Sequential(
            nn.Linear(in_channels_img, out_channels),
            nn.BatchNorm1d(out_channels)
        )
        self.input_pet = nn.Sequential(
            nn.Linear(in_channels_img, out_channels),
            nn.BatchNorm1d(out_channels)
        )
        self.input_demo = nn.Sequential(
            nn.Linear(in_channels_nonimg, out_channels),
            nn.BatchNorm1d(out_channels)
        )

        for m in self.modules():
            # if isinstance(m, conv_type):
            #     nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            # elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            #     nn.init.constant_(torch.as_tensor(m.weight), 1)
            #     nn.init.constant_(torch.as_tensor(m.bias), 0)
            if isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, feats1, feats2, feats3):
        """
            objs : 1xDxN
        """
        # feats = objs.view(2, -1, objs.size(-1))  # 1x2DxL -> 2xDxL
        gate_mri = self.gate_mri(feats1)  # 2xDxL
        gate_pet = self.gate_pet(feats2)  # 2xDxL
        gate_nonimg = self.gate_demo(feats3)  # 2xDxL
        obj_fused = gate_mri.mul(self.input_mri(feats1)) + gate_pet.mul(
            self.input_pet(feats2)) + gate_nonimg.mul(self.input_demo(feats3))

        obj_feats = torch.cat([feats1, feats2, feats3, obj_fused.div(gate_mri + gate_pet + gate_nonimg)], dim=1)
        return obj_feats


class fusion_module_B(nn.Module):

    def __init__(self, appear_len, point_len, out_channels):
        super(fusion_module_B, self).__init__()
        print("Fusion Module B: point, weighted image"
              "& linear fusion, with split input w")
        self.appear_len = appear_len
        self.point_len = point_len
        self.input_p = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1),
            nn.GroupNorm(out_channels, out_channels),
        )
        self.input_i = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1),
            nn.GroupNorm(out_channels, out_channels),
        )

    def forward(self, objs):
        """
            objs : 1xDxN
        """

        feats = objs.view(2, -1, objs.size(-1))  # 1x2DxL -> 2xDxL
        obj_fused = self.input_p(feats[:1]) + self.input_i(feats[1:])
        obj_feats = torch.cat([feats, obj_fused], dim=0)
        return obj_feats


class fusion_module_A(nn.Module):

    def __init__(self, appear_len, point_len, out_channels):
        super(fusion_module_A, self).__init__()
        print("Fusion Module A: concatenate point, image & linear fusion")
        self.appear_len = appear_len
        self.point_len = point_len
        self.input_w = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, 1, 1),
            nn.GroupNorm(out_channels, out_channels),
        )

    def forward(self, objs):
        """
            objs : 1xDxN
        """
        feats = objs.view(2, -1, objs.size(-1))  # 1x2DxL -> 2xDxL
        obj_fused = self.input_w(objs)  # 1x2DxL -> 1xDxL
        obj_feats = torch.cat([feats, obj_fused], dim=0)
        return obj_feats