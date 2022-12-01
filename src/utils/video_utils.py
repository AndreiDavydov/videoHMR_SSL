import tempfile

import cv2
import imageio

import numpy as np
# from iopath.common.file_io import PathManager
# from iopath.fb.manifold import ManifoldPathHandler

# pathmgr = PathManager()
# pathmgr.register_handler(ManifoldPathHandler(), allow_override=True)


def get_frames(vid_file, scale=1):
    """
    Read video frames from a file
    """

    video = cv2.VideoCapture(vid_file)
    if video.isOpened() is False:
        print("Error opening video file.")
        exit()
    fps = video.get(cv2.CAP_PROP_FPS)

    frame_list = []
    while video.isOpened():
        ret, frame = video.read()
        if ret is False:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if scale != 1:
            img = cv2.resize(
                img, (int(img.shape[1] * scale), int(img.shape[0] * scale))
            )
        frame_list.append(img)
    video.release()

    return frame_list, fps


def write_video(res_frames, out_path, fps=30):
    """
    Write frames to a video
    """
    width = res_frames[0].shape[1]
    height = res_frames[0].shape[0]

    tmp_file = out_path
    # tmp_file = tempfile.mkdtemp() + ".mp4"
    # out = cv2.VideoWriter(tmp_file, 0x00000021, fps, (width, height))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_file, fourcc, fps, (width, height))
    for frame in res_frames:
        out.write(frame[:, :, ::-1])
    out.release()

    # pathmgr.copy_from_local(tmp_file, out_path, overwrite=True)
    # print(f"Video is saved to '{out_path}'")
    print(f"Video is saved to '{tmp_file}'")
    


def concat_videos(video_paths, out_path="/tmp/concat_vid.mp4", fps=20, scale=1):
    """
    Stacks all videos frame-by-frame horizontally
    NOTE: All video frames must have the same shapes and #frames!
    """

    vid_frames = []
    for vid_path in video_paths:
        # frame_list, _ = get_frames(pathmgr.get_local_path(vid_path), scale=scale)
        frame_list, _ = get_frames(vid_path, scale=scale)
        vid_frames.append(frame_list)

    num_frames = len(frame_list)

    out_frames = []
    for frame_idx in range(num_frames):
        out_frame = [vid[frame_idx] for vid in vid_frames]
        out_frame = np.concatenate(out_frame, axis=1)
        out_frames.append(out_frame)

    write_video(out_frames, out_path, fps=fps)


def write_video_gif(res_frames, out_path, fps=30):
    """
    Write frames to a video in a gif format
    """

    tmp_file = tempfile.mkdtemp() + ".gif"
    imageio.mimsave(tmp_file, res_frames, format="GIF", fps=fps)

    # pathmgr.copy_from_local(tmp_file, out_path, overwrite=True)
    # print(f"Video is saved to '{out_path}'")
    print(f"Video is saved to '{tmp_file}'")


def test_write_video_gif():
    vid_path = (
        "manifold://xr_body/tree/personal/andreydavydov/eft/sampledata/han_short.mp4"
    )
    # frame_list, _ = get_frames(pathmgr.get_local_path(vid_path))
    frame_list, _ = get_frames(vid_path)
    frame_list = frame_list[::10]  # gif is quite heavy

    vid_path_new = "/tmp/test_gif.gif"
    write_video_gif(frame_list, vid_path_new, fps=20)


def test_concat_videos():
    video_paths = [
        "manifold://xr_body/tree/personal/andreydavydov/eft/sampledata/han_short.mp4",
        "manifold://xr_body/tree/personal/andreydavydov/eft/sampledata/han_short.mp4",
    ]
    concat_videos(video_paths, out_path="/tmp/concat_vid.mp4", scale=2)


if __name__ == "__main__":
    test_concat_videos()

    test_write_video_gif()
