from models.Unet_SA import Generator
import os
import torch
import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy
from PIL import Image


def trans(position, im_height, device):
    im_width = im_height * 2
    image = torch.randn([1, im_height, im_width], device=device)

    for i, pos in enumerate(position):
        if not torch.any(pos):
            continue
        cent_x, cent_y = round(im_width * pos[0].item()), round(im_height * (1 - pos[1].item()))
        image[:, cent_y - 2:cent_y + 2, cent_x - 2:cent_x + 2] = 100

    return image


def transform(positions, im_height, device):
    images = torch.zeros(positions.shape[0], 1, im_height, im_height * 2, device=device)
    for i, pos in enumerate(positions):
        images[i] = trans(pos, im_height, device)

    return images


def show_img(img_tensor, save_path=None):
    plt.imshow(img_tensor)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # 保存图像到指定路径
    else:
        plt.show()


def save_image(img, path):
    img = (img * 255).astype('uint8')
    Image.fromarray(img).save(path)


class Verifier(object):
    def __init__(self, config, n_pth):
        self.config = config
        self.load_model(n_pth)

    def load_model(self, n_pth):
        self.G = Generator(self.config.batch_size, self.config.imsize, self.config.z_dim,
                           self.config.g_conv_dim).to(self.config.device)
        self.G.load_state_dict(
            torch.load(os.path.join(self.config.model_save_path, '{}_G.pth'.format(n_pth)), weights_only=True))
        self.G.eval()
        print('loaded trained models (step: {})..!'.format(n_pth))

    def predict_img_by_pos(self, pos):
        pos_batch = pos.repeat(self.config.batch_size, 1, 1)
        pos2images = transform(pos_batch, self.config.imsize, self.config.device)
        fake_images = self.G(pos2images)
        fake_img = numpy.array(fake_images[0].permute(1, 2, 0).detach().to('cpu'))
        return fake_img


config = SimpleNamespace(
    batch_size=1,
    imsize=64,
    z_dim=1,
    g_conv_dim=64,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    model_save_path='pths\\sagan_1',
)

if __name__ == '__main__':
    # 根据输入的pos，生成对应的风场预测结果
    # predict_pos: 输入的风场位置，最少2行，最多5行，每行2列，表示x,y坐标，范围0-1
    # 确保两栋建筑之间距离（无论是水平距离还是垂直距离）最小为建筑宽度的一半，否则虽然不会报错，但是生成的风场预测结果可能不准确
    V = Verifier(config, 34600)
    predict_pos = torch.tensor([
        [0.5, 0.5],
        [0.38, 0.25],
        [0.7, 0.65],
    ], device=config.device)
    fake_img = V.predict_img_by_pos(predict_pos)
    show_img(fake_img)
