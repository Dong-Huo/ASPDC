from networks.modules import *
import numpy as np

class DeblurringNet(nn.Module):
    def __init__(self, in_channels=3, intermediate_channels=(32, 64, 128, 256), out_channels=3, num_res=2,
                 norm_layer=nn.InstanceNorm2d):
        super(DeblurringNet, self).__init__()

        self.head = nn.Sequential(nn.Conv2d(in_channels, intermediate_channels[0], 3, 1, 1),
                                  nn.LeakyReLU(0.1, inplace=True),
                                  ResBlock(intermediate_channels[0], intermediate_channels[0], norm_layer=norm_layer),
                                  ResBlock(intermediate_channels[0], intermediate_channels[0], norm_layer=norm_layer),
                                  nn.Conv2d(intermediate_channels[0], intermediate_channels[1], 3, 2, 1),
                                  nn.LeakyReLU(0.1, inplace=True),
                                  nn.Conv2d(intermediate_channels[1], intermediate_channels[2], 3, 2, 1),
                                  nn.LeakyReLU(0.1, inplace=True))

        # encoder1
        self.en_res_group1_1 = nn.Sequential(
            *[ResBlock(intermediate_channels[2], intermediate_channels[2], norm_layer=norm_layer) for _ in
              range(num_res)])
        self.en_res_group1_2 = hallucination_res_block(intermediate_channels[2], intermediate_channels[2],
                                                       dilations=(0, 1, 2, 4), norm_layer=norm_layer)

        # encoder2
        self.en_res_group2_1 = nn.Sequential(
            *[ResBlock(intermediate_channels[2], intermediate_channels[2], norm_layer=norm_layer) for _ in
              range(num_res)])
        self.en_res_group2_2 = hallucination_res_block(intermediate_channels[2], intermediate_channels[2],
                                                       dilations=(0, 1, 2, 4), norm_layer=norm_layer)

        # encoder3
        self.en_res_group3_1 = nn.Sequential(
            *[ResBlock(intermediate_channels[2], intermediate_channels[2], norm_layer=norm_layer) for _ in
              range(num_res)])
        self.en_res_group3_2 = hallucination_res_block(intermediate_channels[2], intermediate_channels[2],
                                                       dilations=(0, 1, 2, 4), norm_layer=norm_layer)

        # decoder3

        self.de_res_group3_1 = nn.Sequential(
            *[ResBlock(intermediate_channels[2], intermediate_channels[2], norm_layer=norm_layer) for _ in
              range(num_res)])
        self.de_res_group3_2 = hallucination_res_block(intermediate_channels[2], intermediate_channels[2],
                                                       dilations=(0, 1, 2, 4), norm_layer=norm_layer)

        # decoder2

        self.de_res_group2_1 = nn.Sequential(
            *[ResBlock(intermediate_channels[2], intermediate_channels[2], norm_layer=norm_layer) for _ in
              range(num_res)])
        self.de_res_group2_2 = hallucination_res_block(intermediate_channels[2], intermediate_channels[2],
                                                       dilations=(0, 1, 2, 4), norm_layer=norm_layer)

        # decoder1

        self.de_res_group1_1 = nn.Sequential(
            *[ResBlock(intermediate_channels[2], intermediate_channels[2], norm_layer=norm_layer) for _ in
              range(num_res)])
        self.de_res_group1_2 = hallucination_res_block(intermediate_channels[2], intermediate_channels[2],
                                                       dilations=(0, 1, 2, 4), norm_layer=norm_layer)

        self.dense_conv = nn.Sequential(nn.Conv2d(intermediate_channels[2] * 6, intermediate_channels[2], 3, 1, 1),
                                        nn.LeakyReLU(0.1, inplace=True))

        # self.bridge = Cross_Attn(intermediate_channels[2])

        self.tail = nn.Sequential(
            nn.ConvTranspose2d(intermediate_channels[2], intermediate_channels[1], 3, 2, 1, output_padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(intermediate_channels[1], intermediate_channels[0], 3, 2, 1, output_padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(intermediate_channels[0], out_channels, 3, 1, 1))

    def forward(self, rgb):
        x = self.head(rgb)

        hallucination_map_list = []

        # encoder1
        en_res_out1_1 = self.en_res_group1_1(x)
        en_res_out1_2, en_map1 = self.en_res_group1_2(en_res_out1_1)

        hallucination_map_list.append(en_map1)
        # encoder2
        en_res_out2_1 = self.en_res_group2_1(en_res_out1_2)
        en_res_out2_2, en_map2 = self.en_res_group2_2(en_res_out2_1)

        hallucination_map_list.append(en_map2)

        # encoder3
        en_res_out3_1 = self.en_res_group3_1(en_res_out2_2)
        en_res_out3_2, en_map3 = self.en_res_group3_2(en_res_out3_1)

        hallucination_map_list.append(en_map3)

        # decoder3

        de_res_out3_1 = self.de_res_group3_1(en_res_out3_2)
        de_res_out3_2, de_map3 = self.de_res_group3_2(de_res_out3_1)

        hallucination_map_list.append(de_map3)

        # decoder2

        de_res_out2_1 = self.de_res_group2_1(de_res_out3_2)
        de_res_out2_2, de_map2 = self.de_res_group2_2(de_res_out2_1)

        hallucination_map_list.append(de_map2)

        # decoder1

        de_res_out1_1 = self.de_res_group1_1(de_res_out2_2)
        de_res_out1_2, de_map1 = self.de_res_group1_2(de_res_out1_1)

        hallucination_map_list.append(de_map1)

        dense_output = self.dense_conv(
            torch.cat([en_res_out1_2, en_res_out2_2, en_res_out3_2, de_res_out3_2, de_res_out2_2, de_res_out1_2], 1))

        # bridge
        res = self.tail(dense_output)

        return res + rgb, res

class BlurringNet(nn.Module):
    def __init__(self, in_channels=3, intermediate_channels=(16, 32, 64, 128, 256, 512), num_recurrent=3,
                 intermediate_kernels=3, norm_layer=nn.InstanceNorm2d):
        super(BlurringNet, self).__init__()
        self.num_recurrent = num_recurrent

        # extract features from the blurry image and sharp image
        self.head_conv = nn.Sequential(nn.Conv2d(in_channels * 2, intermediate_channels[0], 3, 1, 1),
                                       nn.LeakyReLU(0.1, inplace=True))

        self.kernel_head_conv = nn.Conv2d(intermediate_channels[0], np.square(intermediate_kernels), 3, 1, 1)

        self.body_conv_1 = nn.Sequential(nn.Conv2d(intermediate_channels[0], intermediate_channels[1], 3, 2, 1),
                                         ResBlock(intermediate_channels[1], intermediate_channels[1],
                                                  norm_layer=norm_layer),
                                         # ResBlock(intermediate_channels[1], intermediate_channels[1],
                                         #          norm_layer=norm_layer),
                                         nn.LeakyReLU(0.1, inplace=True))

        self.kernel_conv_1 = nn.Conv2d(intermediate_channels[1], np.square(intermediate_kernels), 3, 1, 1)

        self.body_conv_2 = nn.Sequential(nn.Conv2d(intermediate_channels[1], intermediate_channels[2], 3, 2, 1),
                                         ResBlock(intermediate_channels[2], intermediate_channels[2],
                                                  norm_layer=norm_layer),
                                         # ResBlock(intermediate_channels[2], intermediate_channels[2],
                                         #          norm_layer=norm_layer),

                                         nn.LeakyReLU(0.1, inplace=True))

        self.kernel_conv_2 = nn.Conv2d(intermediate_channels[2], np.square(intermediate_kernels), 3, 1, 1)

        self.body_conv_3 = nn.Sequential(nn.Conv2d(intermediate_channels[2], intermediate_channels[3], 3, 2, 1),
                                         ResBlock(intermediate_channels[3], intermediate_channels[3],
                                                  norm_layer=norm_layer),
                                         # ResBlock(intermediate_channels[3], intermediate_channels[3],
                                         #          norm_layer=norm_layer),
                                         nn.LeakyReLU(0.1, inplace=True))

        self.kernel_conv_3 = nn.Conv2d(intermediate_channels[3], np.square(intermediate_kernels), 3, 1, 1)

        self.body_dconv_3 = nn.Sequential(ResBlock(intermediate_channels[3], intermediate_channels[3],
                                                   norm_layer=norm_layer),
                                          # ResBlock(intermediate_channels[3], intermediate_channels[3],
                                          #          norm_layer=norm_layer),
                                          nn.ConvTranspose2d(intermediate_channels[3], intermediate_channels[2], 3, 2,
                                                             1, output_padding=1),
                                          nn.LeakyReLU(0.1, inplace=True))

        self.kernel_dconv_3 = nn.Conv2d(intermediate_channels[2], np.square(intermediate_kernels), 3, 1, 1)

        self.body_dconv_2 = nn.Sequential(ResBlock(intermediate_channels[2], intermediate_channels[2],
                                                   norm_layer=norm_layer),
                                          # ResBlock(intermediate_channels[2], intermediate_channels[2],
                                          #          norm_layer=norm_layer),
                                          nn.ConvTranspose2d(intermediate_channels[2], intermediate_channels[1], 3, 2,
                                                             1, output_padding=1),
                                          nn.LeakyReLU(0.1, inplace=True))

        self.kernel_dconv_2 = nn.Conv2d(intermediate_channels[1], np.square(intermediate_kernels), 3, 1, 1)

        self.body_dconv_1 = nn.Sequential(ResBlock(intermediate_channels[1], intermediate_channels[1],
                                                   norm_layer=norm_layer),
                                          # ResBlock(intermediate_channels[1], intermediate_channels[1],
                                          #          norm_layer=norm_layer),
                                          nn.ConvTranspose2d(intermediate_channels[1], intermediate_channels[0], 3, 2,
                                                             1, output_padding=1),
                                          nn.LeakyReLU(0.1, inplace=True))

        self.kernel_dconv_1 = nn.Conv2d(intermediate_channels[0], np.square(intermediate_kernels), 3, 1, 1)

        # four kernels with different size

        self.tail = nn.Conv2d(intermediate_channels[0], in_channels, 3, 1, 1)

        self.dynamic_conv = Dynamic_conv(intermediate_kernels)

    def forward(self, blurry_image, sharp_image):
        fusion_feature = self.head_conv(torch.cat([blurry_image, sharp_image], 1))

        head_kernel = self.kernel_head_conv(fusion_feature)

        body_out_1 = self.body_conv_1(fusion_feature)
        kernel_1 = self.kernel_conv_1(body_out_1)

        body_out_2 = self.body_conv_2(body_out_1)
        kernel_2 = self.kernel_conv_2(body_out_2)

        body_out_3 = self.body_conv_3(body_out_2)
        kernel_3 = self.kernel_conv_3(body_out_3)

        body_dout_3 = self.body_dconv_3(body_out_3) + body_out_2
        kernel_d3 = self.kernel_dconv_3(body_dout_3)

        body_dout_2 = self.body_dconv_2(body_dout_3) + body_out_1
        kernel_d2 = self.kernel_dconv_2(body_dout_2)

        body_dout_1 = self.body_dconv_1(body_dout_2)
        kernel_d1 = self.kernel_dconv_1(body_dout_1)
        # ----------------------------------------------------------------
        fusion_feature = self.head_conv(torch.cat([sharp_image, sharp_image], 1))
        fusion_feature = self.dynamic_conv(fusion_feature, head_kernel)

        body_out_1 = self.body_conv_1(fusion_feature)
        body_out_1 = self.dynamic_conv(body_out_1, kernel_1)

        body_out_2 = self.body_conv_2(body_out_1)
        body_out_2 = self.dynamic_conv(body_out_2, kernel_2)

        body_out_3 = self.body_conv_3(body_out_2)
        body_out_3 = self.dynamic_conv(body_out_3, kernel_3)

        body_dout_3 = self.body_dconv_3(body_out_3) + body_out_2
        body_dout_3 = self.dynamic_conv(body_dout_3, kernel_d3)

        body_dout_2 = self.body_dconv_2(body_dout_3) + body_out_1
        body_dout_2 = self.dynamic_conv(body_dout_2, kernel_d2)

        body_dout_1 = self.body_dconv_1(body_dout_2)
        body_dout_1 = self.dynamic_conv(body_dout_1, kernel_d1)

        output = self.tail(body_dout_1)

        return output

