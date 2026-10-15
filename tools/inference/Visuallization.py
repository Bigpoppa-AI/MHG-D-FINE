"""
VisDrone Inference & Visualization Script
Based on D-FINE / DEIM framework
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
import os
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


# ============================================================
# VisDrone 类别映射（兼容 0-indexed 和 1-indexed 两种情况）
# ============================================================
VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

# 两种索引方式都建一份，推理时自动适配
label_map_0idx = {i: name for i, name in enumerate(VISDRONE_CLASSES)}        # 0~9
label_map_1idx = {i + 1: name for i, name in enumerate(VISDRONE_CLASSES)}    # 1~10


# ============================================================
# 配色方案：10 个类别 × 高对比度颜色（适合无人机俯视图）
# ============================================================
# 使用 tab10 配色，每类一个固定颜色，论文/PPT 都好看
_palette = plt.cm.tab10.colors
COLOR_MAP = {
    i: tuple(int(c * 255) for c in _palette[i % len(_palette)])
    for i in range(11)  # 0~10 都覆盖
}


# ============================================================
# 绘制函数（针对 VisDrone 小目标做了优化）
# ============================================================
def draw_visdrone(image, labels, boxes, scores, thrh=0.3, label_map=None):
    """
    针对 VisDrone 小目标场景优化的可视化：
    - 更细的边框（width=2）避免遮挡小目标
    - 文字放在框上方，背景半透明
    - 字体稍小以适应密集场景
    """
    drawer = ImageDraw.Draw(image, 'RGBA')

    # 尝试加载稍大的字体（如果系统有的话），否则用默认
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    mask = scores > thrh
    labels = labels[mask]
    boxes = boxes[mask]
    scores = scores[mask]

    if label_map is None:
        label_map = label_map_0idx

    for j, box in enumerate(boxes):
        cat_id = int(labels[j].item())
        cat_name = label_map.get(cat_id, f"cls_{cat_id}")
        color = COLOR_MAP.get(cat_id, (255, 255, 255))

        x1, y1, x2, y2 = map(int, box)

        # 画框（width=2 适合小目标）
        drawer.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 文字内容
        text = f"{cat_name} {scores[j].item():.2f}"
        text_bbox = drawer.textbbox((0, 0), text, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # 文字背景（半透明）
        bg_y1 = max(0, y1 - th - 4)
        bg = [x1, bg_y1, x1 + tw + 6, y1]
        drawer.rectangle(bg, fill=(*color, 200))  # alpha=200
        drawer.text((x1 + 3, bg_y1 + 1), text, fill="white", font=font)

    return image


# ============================================================
# 解析模型输出（兼容 tuple / dict 两种格式）
# ============================================================
def parse_output(output):
    if isinstance(output, (list, tuple)) and len(output) == 3:
        labels, boxes, scores = output
        return labels[0], boxes[0], scores[0]
    elif isinstance(output, list) and isinstance(output[0], dict):
        return output[0]['labels'], output[0]['boxes'], output[0]['scores']
    else:
        raise ValueError(f"Unsupported output format: {type(output)}")


# ============================================================
# 自动检测类别索引方式（0-indexed or 1-indexed）
# ============================================================
def detect_label_map(labels_tensor):
    """根据预测的标签范围自动判断用哪套 label_map"""
    if labels_tensor.numel() == 0:
        return label_map_0idx
    max_id = int(labels_tensor.max().item())
    min_id = int(labels_tensor.min().item())
    # 如果出现 10 或最小是 1，倾向 1-indexed
    if max_id >= 10 or min_id >= 1:
        return label_map_1idx
    return label_map_0idx


# ============================================================
# 单张图片推理
# ============================================================
def infer_single_image(model, file_path, output_dir, thrh=0.3):
    os.makedirs(output_dir, exist_ok=True)
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).cuda()

    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    im_data = transforms(im_pil).unsqueeze(0).cuda()

    output = model(im_data, orig_size)
    labels, boxes, scores = parse_output(output)
    lmap = detect_label_map(labels)

    vis = draw_visdrone(im_pil.copy(), labels, boxes, scores, thrh, lmap)
    save_path = os.path.join(output_dir, f"vis_{os.path.basename(file_path)}")
    vis.save(save_path)
    print(f"✅ 检测到 {(scores > thrh).sum().item()} 个目标，已保存：{save_path}")


# ============================================================
# 批量处理 VisDrone 测试集文件夹
# ============================================================
def process_dataset(model, dataset_path, output_path, thrh=0.3, max_num=None):
    os.makedirs(output_path, exist_ok=True)
    image_paths = sorted([
        os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    if max_num is not None:
        image_paths = image_paths[:max_num]

    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    print(f"📂 在 {dataset_path} 找到 {len(image_paths)} 张图片")
    for idx, file_path in enumerate(image_paths):
        im_pil = Image.open(file_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).cuda()
        im_data = transforms(im_pil).unsqueeze(0).cuda()

        with torch.no_grad():
            output = model(im_data, orig_size)
        labels, boxes, scores = parse_output(output)
        lmap = detect_label_map(labels)

        vis = draw_visdrone(im_pil.copy(), labels, boxes, scores, thrh, lmap)
        save_path = os.path.join(output_path, f"vis_{os.path.basename(file_path)}")
        vis.save(save_path)

        if (idx + 1) % 50 == 0:
            print(f"  已处理 {idx + 1}/{len(image_paths)} 张")

    print(f"✅ 可视化完成，结果保存在：{output_path}")


# ============================================================
# 视频推理（无人机视频很常见）
# ============================================================
def process_video(model, file_path, output_dir, thrh=0.3):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_dir, f"vis_{os.path.basename(file_path)}")
    out = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    frame_count = 0
    print(f"🎬 开始处理视频：{file_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).cuda()
        im_data = transforms(frame_pil).unsqueeze(0).cuda()

        with torch.no_grad():
            output = model(im_data, orig_size)
        labels, boxes, scores = parse_output(output)
        lmap = detect_label_map(labels)

        vis = draw_visdrone(frame_pil, labels, boxes, scores, thrh, lmap)
        frame_out = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        out.write(frame_out)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  已处理 {frame_count} 帧")

    cap.release()
    out.release()
    print(f"✅ 视频保存：{out_path}")


# ============================================================
# 主函数
# ============================================================
def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.eval().cuda()
            self.postprocessor = cfg.postprocessor.eval().cuda()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model()

    # 分支逻辑
    if args.input:
        ext = os.path.splitext(args.input)[-1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            infer_single_image(model, args.input, args.output, thrh=args.thrh)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            process_video(model, args.input, args.output, thrh=args.thrh)
        else:
            raise ValueError(f"不支持的输入格式：{ext}")
    elif args.dataset:
        process_dataset(model, args.dataset, args.output,
                        thrh=args.thrh, max_num=args.max_num)
    else:
        raise ValueError("请通过 --input 或 --dataset 指定输入！")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="VisDrone 推理与可视化")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="训练时使用的 yaml 配置文件")
    parser.add_argument('-r', '--resume', type=str, required=True,
                        help="模型权重路径 (.pth)")
    parser.add_argument('-i', '--input', type=str, default=None,
                        help="单张图片或视频路径")
    parser.add_argument('-d', '--dataset', type=str, default=None,
                        help="图片文件夹路径（批量处理 VisDrone test-dev）")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="可视化结果保存目录")
    parser.add_argument('-t', '--thrh', type=float, default=0.3,
                        help="置信度阈值（VisDrone 小目标建议 0.2~0.3）")
    parser.add_argument('--max_num', type=int, default=None,
                        help="最多处理多少张图（用于快速调试）")
    args = parser.parse_args()
    main(args)
