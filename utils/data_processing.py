# utils/data_processing.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class DataAugmentation:
    @staticmethod
    def preprocess_frame(frame, input_size=112):
        """统一的预处理方法"""
        # 确保帧尺寸大于或等于输入尺寸
        h, w, _ = frame.shape
        if h < input_size or w < input_size:
            scale_factor = max(input_size / h, input_size / w)
            frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)

        # 中心裁剪或填充
        h, w, _ = frame.shape
        if h > input_size or w > input_size:
            x = (w - input_size) // 2
            y = (h - input_size) // 2
            frame = frame[y:y + input_size, x:x + input_size, :]
        else:
            padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
            x = (input_size - w) // 2
            y = (input_size - h) // 2
            padded[y:y + h, x:x + w, :] = frame[:h, :w, :]
            frame = padded

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    @staticmethod
    def rotate(frame, angle):
        """旋转图像"""
        (h, w) = frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h))
        return rotated

    @staticmethod
    def flip(frame, axis=0):
        """翻转图像"""
        flipped = cv2.flip(frame, axis)
        return flipped

    @staticmethod
    def crop(frame, ratio=0.85):
        """裁剪图像"""
        h, w, _ = frame.shape
        crop_h = int(h * ratio)
        crop_w = int(w * ratio)
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2
        cropped = frame[y:y + crop_h, x:x + crop_w, :]
        cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)
        return cropped

    @staticmethod
    def add_noise(frame, mean=0, var=10):
        """添加噪声"""
        gauss = np.random.normal(mean, var ** 0.5, frame.shape).astype(np.uint8)
        noisy = cv2.add(frame, gauss)
        return noisy

    @staticmethod
    def scale(frame, scale_factor=1.1):
        """缩放图像并中心裁剪或调整尺寸"""
        h, w, _ = frame.shape
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 如果缩放后的图像比原图大，则中心裁剪
        if new_h > h or new_w > w:
            x = (new_w - w) // 2
            y = (new_h - h) // 2
            scaled = scaled[y:y + h, x:x + w, :]
        else:
            # 如果缩放后的图像比原图小，则调整到原图尺寸
            scaled = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)

        return scaled

    @staticmethod
    def color_jitter(frame):
        """颜色抖动"""
        frame = frame.astype(np.float32)
        brightness_factor = np.random.uniform(0.8, 1.2)
        contrast_factor = np.random.uniform(0.8, 1.2)
        saturation_factor = np.random.uniform(0.8, 1.2)
        hue_factor = np.random.uniform(-0.1, 0.1)
        frame *= brightness_factor
        frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        frame[:, :, 1] = np.clip(frame[:, :, 1] * saturation_factor, 0, 255)
        frame[:, :, 0] = (frame[:, :, 0].astype(int) + np.deg2rad(hue_factor) * 180 / np.pi) % 180
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    @staticmethod
    def adjust_brightness(frame, factor=1.2):
        """调整亮度"""
        frame = cv2.convertScaleAbs(frame, alpha=factor, beta=0)
        return frame

    @staticmethod
    def adjust_contrast(frame, factor=1.2):
        """调整对比度"""
        mean = np.mean(frame)
        frame = np.clip((factor * frame - factor * mean + mean), 0, 255).astype(np.uint8)
        return frame

    @staticmethod
    def grayscale(frame):
        """灰度化"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return gray

    @staticmethod
    def hue_shift(frame, delta=10):
        """色相偏移"""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        frame[:, :, 0] = (frame[:, :, 0].astype(int) + delta) % 180
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
        return frame


class FruitVideoDataset(Dataset):
    def __init__(self, data_dir, model_name="i3d", num_frames=16, input_size=112,
                 train=True, augment=False, target_count=None):
        self.data_dir = data_dir
        self.model_name = model_name
        self.num_frames = num_frames
        self.input_size = input_size
        self.train = train
        self.augment = augment
        self.target_count = target_count

        self.classes = {"好果": 0, "次果": 1, "烂果": 2}
        self.class_names = list(self.classes.keys())

        self.video_paths, self.labels = self._load_data()

        # 如果需要增强数据
        if self.train and self.augment and self.target_count:
            self.video_paths, self.labels = self._augment_data()

    def _load_data(self):
        video_paths = []
        labels = []

        if not os.path.exists(self.data_dir):
            print(f"⚠️ 警告：目录不存在 {self.data_dir}")
            return [], []

        print(f"正在加载数据集目录: {self.data_dir}")

        for cls_name, cls_idx in self.classes.items():
            cls_dir = os.path.join(self.data_dir, cls_name)
            if not os.path.exists(cls_dir):
                print(f"⚠️ 警告：类别目录不存在 {cls_dir}")
                continue

            print(f"正在加载类别目录: {cls_dir}")

            for video_file in os.listdir(cls_dir):
                if video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_paths.append(os.path.join(cls_dir, video_file))
                    labels.append(cls_idx)

        total = len(video_paths)
        counts = [labels.count(i) for i in range(len(self.classes))]
        print(f"\n{'训练' if self.train else '验证'}集加载完成：{total}个视频")
        for i, name in enumerate(self.class_names):
            print(f"  {name}: {counts[i]}个")

        return video_paths, labels

    def _augment_data(self):
        """增强数据以达到目标数量"""
        if not self.target_count:
            return self.video_paths, self.labels

        augmented_paths = []
        augmented_labels = []

        class_data = {i: [] for i in range(3)}
        for path, label in zip(self.video_paths, self.labels):
            class_data[label].append((path, label))

        for cls_idx, paths in class_data.items():
            current_count = len(paths)
            needed = self.target_count - current_count
            print(f"\n【{self.class_names[cls_idx]}】原始: {current_count}, 需要增强: {needed}个")

            augmented_paths.extend([p for p, _ in paths])
            augmented_labels.extend([cls_idx] * current_count)

            if needed > 0:
                for i in range(needed):
                    if not paths:  # 确保路径列表不为空
                        print(f"⚠️ 警告：类别 {self.class_names[cls_idx]} 没有足够的视频进行增强")
                        break
                    path, _ = random.choice(paths)
                    aug_type = random.randint(0, 6)  # 更新为0-6
                    augmented_paths.append(f"{path}||aug_{aug_type}")
                    augmented_labels.append(cls_idx)

        final_counts = [augmented_labels.count(i) for i in range(3)]
        print("\n增强后数据分布:")
        for i, name in enumerate(self.class_names):
            print(f"  {name}: {final_counts[i]}个")

        return augmented_paths, augmented_labels

    def _load_video_frames(self, video_path):
        """加载视频帧并应用增强"""
        aug_type = None
        if "||aug_" in video_path:
            video_path, aug_info = video_path.split("||")
            aug_type = int(aug_info.split("_")[1])

        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print(f"⚠️ 警告：视频文件损坏或无帧 {video_path}")
            return torch.zeros(self.num_frames, 3, self.input_size, self.input_size)

        if total_frames < self.num_frames:
            # 使用插值方法填充帧
            sample_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=float)
            prev_frame = None
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frame = DataAugmentation.preprocess_frame(frame, self.input_size)
                    if aug_type is not None:
                        frame = self.apply_augmentation(frame, aug_type)
                    if self.train and aug_type is None:
                        frame = self.apply_random_augmentations(frame)
                    frames.append(frame)
                    prev_frame = frame
                else:
                    if prev_frame is not None:
                        # 插值生成新帧
                        new_frame = self.interpolate_frames(prev_frame, prev_frame, self.input_size)
                        frames.append(new_frame)
                    else:
                        frames.append(np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8))
        else:
            sample_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = DataAugmentation.preprocess_frame(frame, self.input_size)
                    if aug_type is not None:
                        frame = self.apply_augmentation(frame, aug_type)
                    if self.train and aug_type is None:
                        frame = self.apply_random_augmentations(frame)
                    frames.append(frame)
                else:
                    frames.append(np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8))

        cap.release()

        # 确保所有帧的形状一致
        frames = [DataAugmentation.preprocess_frame(frame, self.input_size) for frame in frames]

        frames = np.array(frames, dtype=np.uint8)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(3, 0, 1, 2)

        if self.train and self.model_name.lower() in ["transformer", "mamba"] and aug_type is None:
            if frames.ndim == 4 and frames.shape[1] > 1:
                t = np.random.randint(0, frames.shape[1])
                frames[:, t] = torch.zeros_like(frames[:, t])
        return frames

    def interpolate_frames(self, frame1, frame2, input_size):
        """线性插值生成新帧"""
        frame1 = DataAugmentation.preprocess_frame(frame1, input_size)
        frame2 = DataAugmentation.preprocess_frame(frame2, input_size)
        alpha = np.random.rand()
        interpolated_frame = cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)
        return interpolated_frame

    def apply_augmentation(self, frame, aug_type):
        if aug_type == 0:
            frame = DataAugmentation.rotate(frame, 90)
        elif aug_type == 1:
            frame = DataAugmentation.rotate(frame, 180)
        elif aug_type == 2:
            frame = DataAugmentation.rotate(frame, 270)
        elif aug_type == 3:
            frame = DataAugmentation.flip(frame, 1)
        elif aug_type == 4:
            frame = DataAugmentation.crop(frame, 0.85)
        elif aug_type == 5:
            frame = DataAugmentation.add_noise(frame)
        elif aug_type == 6:
            frame = DataAugmentation.scale(frame)
        # 确保帧的尺寸一致
        frame = DataAugmentation.preprocess_frame(frame, self.input_size)
        return frame

    def apply_random_augmentations(self, frame):
        if np.random.rand() > 0.5:
            frame = DataAugmentation.color_jitter(frame)
        if np.random.rand() > 0.7:
            frame = DataAugmentation.flip(frame)
        if np.random.rand() > 0.8:
            frame = DataAugmentation.adjust_brightness(frame)
        if np.random.rand() > 0.6:
            frame = DataAugmentation.adjust_contrast(frame)
        if np.random.rand() > 0.5:
            frame = DataAugmentation.grayscale(frame)
        if np.random.rand() > 0.4:
            frame = DataAugmentation.hue_shift(frame)
        # 确保帧的尺寸一致
        frame = DataAugmentation.preprocess_frame(frame, self.input_size)
        return frame

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        frames = self._load_video_frames(self.video_paths[idx])
        return frames, self.labels[idx]
