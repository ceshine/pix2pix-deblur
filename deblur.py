import sys

import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

from blur_dataset import INVERSE_NORMALIZE

PATCH_SIZE = 128
STRIDE = 110
TO_PIL = transforms.ToPILImage()

if __name__ == "__main__":
    image_name = sys.argv[1]
    netG = torch.load("checkpoint/netG_model_epoch_136.pth").cuda()
    img = Image.open(image_name)
    img_tensor = transforms.ToTensor()(img).cuda()
    if max(img.size) > 512:
        new_im = Image.new('RGB', img.size)
        for i in range(0, img_tensor.size()[1], STRIDE):
            for j in range(0, img_tensor.size()[2], STRIDE):
                patch = Variable(
                    img_tensor[
                        :,
                        i:(i + PATCH_SIZE),
                        j:(j + PATCH_SIZE)
                    ], volatile=True
                ).unsqueeze(0)
                if patch.size()[2] % 2 == 1:
                    patch = patch[:, :, :-1, :]
                if patch.size()[3] % 2 == 1:
                    patch = patch[:, :, :, :-1]
                print(i, j, patch.size())
                # result_tensor = netG(patch).squeeze(0).data[:, 8:-8, 8:-8].cpu()
                result_tensor = (
                    INVERSE_NORMALIZE(netG(patch).data.squeeze(0)) * 255).clamp(0, 255).byte(
                )[:, 8:-8, 8:-8].cpu()
                if result_tensor.size()[1] == 0 or result_tensor.size()[2] == 0:
                    continue
                # result_tensor = (result_tensor * 255).round().byte()
                result_img = TO_PIL(result_tensor)
                new_im.paste(result_img, (j + 8, i + 8))
        new_im.save("output.jpg")
    else:
        result_tensor = (
            INVERSE_NORMALIZE(
                netG(Variable(img_tensor, volatile=True).unsqueeze(0)).data.squeeze(0)
            ) * 255).clamp(0, 255).byte(
        ).cpu()
        result_img = TO_PIL(result_tensor)
        result_img.save("output.jpg")
