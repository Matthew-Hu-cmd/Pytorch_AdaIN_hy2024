import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from model import Model


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
    mask = mask.convert('L')
    height, width = foreground.shape[2], foreground.shape[3]
    mask = mask.resize((width, height))
    mask_tensor = transforms.ToTensor()(mask).to(foreground.device)
    mask_tensor = mask_tensor.expand_as(foreground)

    if foreground.shape != background.shape:
        background = torch.nn.functional.interpolate(background, size=(height, width), mode='bilinear', align_corners=False)

    result = foreground * mask_tensor + background * (1 - mask_tensor)
    return result

def process_image(content_path, style_path, mask_path, output_name, model, device, alpha=1):
    c = Image.open(content_path)
    s = Image.open(style_path)

    # 确保内容图像和风格图像都是三通道的
    c = ensure_three_channels(c)
    s = ensure_three_channels(s)

    c_tensor = trans(c).unsqueeze(0).to(device)
    s_tensor = trans(s).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.generate(c_tensor, s_tensor, alpha)
    
    out = denorm(out, device)

    # Save the generated image
    save_image(out, f'{output_name}.jpg', nrow=1)
    o = Image.open(f'{output_name}.jpg')

    # If a mask is provided, apply it
    if mask_path:
        mask = Image.open(mask_path)
        mask = ensure_three_channels(mask)  # 处理掩码图像为三通道

        s_tensor_resized = trans(s).unsqueeze(0).to(device)
        o_tensor = trans(o).unsqueeze(0).to(device)
        
        final_output = apply_mask(o_tensor, s_tensor_resized, mask)
        final_output = denorm(final_output.squeeze(0), device)

        save_image(final_output, f'{output_name}_masked.jpg')
        print(f'Masked result saved as {output_name}_masked.jpg')

def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--content_dir', '-c', type=str, required=True,
                        help='Directory containing content images')
    parser.add_argument('--style_dir', '-s', type=str, required=True,
                        help='Directory containing style images')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Directory for saving generated images')
    parser.add_argument('--mask_dir', '-m', type=str, default=None,
                        help='Directory containing mask images')
    parser.add_argument('--alpha', '-a', type=float, default=1,
                        help='alpha control the fusion degree in Adain')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--model_state_path', type=str, default='model_state.pth',
                        help='Path to the model state file')

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

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # List content images
    content_images = sorted(os.listdir(args.content_dir))
    style_images = sorted(os.listdir(args.style_dir))
    mask_images = sorted(os.listdir(args.mask_dir)) if args.mask_dir else None

    for idx, content_image in enumerate(content_images):
        content_path = os.path.join(args.content_dir, content_image)
        style_path = os.path.join(args.style_dir, style_images[idx])
        mask_path = os.path.join(args.mask_dir, mask_images[idx]) if mask_images else None
        output_name = os.path.join(args.output_dir, os.path.splitext(content_image)[0])

        process_image(content_path, style_path, mask_path, output_name, model, device, alpha=args.alpha)

        print(f'Processed {content_image} and saved result to {output_name}.jpg')

if __name__ == '__main__':
    main()