import cv2

def get_video_total_frames(video_path):
    """
    获取视频的实际总帧数
    :param video_path: 视频文件路径（绝对路径/相对路径都可以）
    :return: 总帧数（int）/ 错误提示（str）
    """
    # 打开视频流
    cap = cv2.VideoCapture(video_path)
    # 校验视频是否成功打开
    if not cap.isOpened():
        return f"错误：无法打开视频文件，请检查路径是否正确或文件是否损坏 → {video_path}"
    # 获取总帧数（cv2.CAP_PROP_FRAME_COUNT 是固定参数，代表帧总数）
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 释放视频流（必须执行，避免占用内存）
    cap.release()
    return total_frames

# ------------------- 调用示例 -------------------
if __name__ == "__main__":
    # 替换成你的视频路径，比如 "D:/video/test.mp4" 或 "./demo.mov"
    VIDEO_PATH = "D:/Desktop/水果采样数据/1.24/h/h2.mp4"
    frames = get_video_total_frames(VIDEO_PATH)
    print(f"视频总帧数：{frames}")