import numpy as np

l_shoulder, r_shoulder, l_hip, r_hip = 11, 12, 23, 24

def normalize_body(sequence):
    lm = sequence.reshape(len(sequence), 33, 4).copy()
    xyz = lm[..., :3]
    hip = (xyz[:, l_hip] + xyz[:, r_hip]) / 2
    shoulder = (xyz[:, l_shoulder] + xyz[:, r_shoulder]) / 2

    torso = np.median(np.linalg.norm((shoulder - hip)[:, :2], axis=1))
    xyz -= hip[:, None, :]
    xyz /= torso + 1e-6
    return lm.reshape(len(sequence), 132)