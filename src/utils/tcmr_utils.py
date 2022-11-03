### adapted from https://github.com/hongsukchoi/TCMR_RELEASE/
import numpy as np
import torch
from skimage.util.shape import view_as_windows


def prepare_tcmr_input_features(features_in, seq_len):
    """
    Args:
        features_in : N x F, N - number of frames in a sequence
        seq_len (int) : length of the sequence.
    Out:
        features_out : N x seqlen x F - input features to the temporal encoder of TCMR
    """
    features_out = []
    for curr_idx in range(len(features_in)):
        start_idx = curr_idx - seq_len // 2
        end_idx = curr_idx + seq_len // 2
        if start_idx < 0 or end_idx > len(features_in):
            # repeat the same features "seq_len" times
            feature = features_in[None, curr_idx : curr_idx + 1]
            feature = feature.repeat(1, seq_len, 1)
        else:
            # take features around the curr_idx (seq_len in total)
            feature = features_in[None, start_idx:end_idx]
        features_out.append(feature)

    features_out = torch.cat(features_out, dim=0)
    return features_out


def split_into_chunks(vid_names, seqlen, stride, is_train=True, match_vibe=True):
    video_start_end_indices = []
    video_names, group = np.unique(vid_names, return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])
    # import pdb; pdb.set_trace()
    for idx in range(len(video_names)):
        indexes = indices[idx]
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen,), step=stride)
        start_finish = chunks[:, (0, -1)].tolist()
        if stride != seqlen:
            if match_vibe:
                vibe_chunks = view_as_windows(indexes, (16,), step=16)
                for j in range(1, len(start_finish) + 1):
                    if start_finish[-j][-1] == vibe_chunks[-1][-1]:
                        if j != 1:
                            start_finish = start_finish[: -j + 1]
                        break

            d = start_finish[0][0]
            for j in range(int(seqlen / 2)):
                if is_train:
                    dummy = start_finish[0]
                else:
                    dummy = [d + j, d + j]
                start_finish.insert(j, dummy)
            d = start_finish[-1][0]
            for j in range(int(seqlen / 2 + 0.5) - 1):
                if is_train:
                    dummy = start_finish[-1]
                else:
                    dummy = [d + int(seqlen / 2) + j + 1, d + int(seqlen / 2) + j + 1]
                start_finish.append(dummy)
        video_start_end_indices += start_finish

    return video_start_end_indices


def get_sequence(start_index, end_index, seqlen=16):
    if start_index != end_index:
        return list(range(start_index, end_index + 1))
    else:
        return [start_index for _ in range(seqlen)]
