import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from model import Model

'''
    用这个代码能处理只有灰度图的情况
'''
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])

def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def ensure_three_channels(image):
    """
    确保图像具有三个通道。
    如果图像是灰度图像（一个通道），则将其重复三次，转换为三通道图像。
    """
    if image.mode == 'L':
        # 如果图像是灰度图像，将其转换为三通道（RGB）
        image = image.convert('RGB')
    return image

def apply_mask(foreground, background, mask):
    """
    将前景图像叠加在背景图像上，使用给定的 mask。
    mask 的值应该在 [0, 1] 范围，表示透明度。
    """
    mask = mask.convert('L')  # 将 mask 转为灰度图
    height, width = foreground.shape[2], foreground.shape[3]  # 获取 tensor 的宽度和高度
    mask = mask.resize((width, height))  # 将 mask 的尺寸调整为前景图的大小
    mask_tensor = transforms.ToTensor()(mask).to(foreground.device)  # 转为 tensor
    mask_tensor = mask_tensor.expand_as(foreground)  # 扩展为与前景图相同的形状

    # 确保背景图像和前景图像的尺寸匹配
    if foreground.shape != background.shape:
        # 调整背景图像的大小，使其与前景图像匹配
        background = torch.nn.functional.interpolate(background, size=(height, width), mode='bilinear', align_corners=False)

    # 通过 mask 将前景和背景合并，保留前景的区域
    result = foreground * mask_tensor + background * (1 - mask_tensor)
    return result


def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--content', '-c', type=str, default=None,
                        help='Content image path e.g. content.jpg')
    parser.add_argument('--style', '-s', type=str, default=None,
                        help='Style image path e.g. image.jpg')
    parser.add_argument('--output_name', '-o', type=str, default=None,
                        help='Output path for generated image, no need to add ext, e.g. out')
    parser.add_argument('--alpha', '-a', type=float, default=1,
                        help='alpha control the fusion degree in Adain')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--model_state_path', type=str, default='model_state.pth',
                        help='Path to the model state file')
    parser.add_argument('--mask', '-m', type=str, default=None,
                        help='Path to mask image (must match content image size)')

    args = parser.parse_args()

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    # set model
    model = Model()
    if args.model_state_path is not None:
        model.load_state_dict(torch.load(args.model_state_path, map_location=lambda storage, loc: storage))
    model = model.to(device)

    c = Image.open(args.content)
    s = Image.open(args.style)

    # 确保图像是三通道的
    c = ensure_three_channels(c)
    s = ensure_three_channels(s)

    c_tensor = trans(c).unsqueeze(0).to(device)
    s_tensor = trans(s).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.generate(c_tensor, s_tensor, args.alpha)
    
    out = denorm(out, device)

    if args.output_name is None:
        c_name = os.path.splitext(os.path.basename(args.content))[0]
        s_name = os.path.splitext(os.path.basename(args.style))[0]
        args.output_name = f'{c_name}_{s_name}'

    # Save the generated image
    save_image(out, f'{args.output_name}.jpg', nrow=1)
    o = Image.open(f'{args.output_name}.jpg')

    # If a mask is provided, apply it
    if args.mask:
        mask = Image.open(args.mask)
        mask = ensure_three_channels(mask)  # 处理掩码图像

        s_tensor_resized = trans(s).unsqueeze(0).to(device)
        o_tensor = trans(o).unsqueeze(0).to(device)

        # Apply mask to the generated image and style image
        final_output = apply_mask(o_tensor, s_tensor_resized, mask)
        final_output = denorm(final_output.squeeze(0), device)

        # Save the final image with mask applied
        save_image(final_output, f'{args.output_name}_masked.jpg')
        print(f'Masked result saved as {args.output_name}_masked.jpg')

    print(f'Result saved into files starting with {args.output_name}')


if __name__ == '__main__':
    main()