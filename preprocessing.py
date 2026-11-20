import numpy as np

l_shoulder, r_shoulder, l_hip, r_hip = 11, 12, 23, 24
aspect = 16 / 9

def normalize_body(sequence):
    lm = sequence.reshape(len(sequence), 33, 4).copy()
    xyz = lm[..., :3]
    hip = (xyz[:, l_hip] + xyz[:, r_hip]) / 2
    shoulder = (xyz[:, l_shoulder] + xyz[:, r_shoulder]) / 2

    torso = np.linalg.norm((shoulder - hip)[:, :2], axis=1).max()
    xyz -= hip[:, None, :]
    xyz /= torso + 1e-6
    return lm.reshape(len(sequence), 132)

def hip_knee_gap(normalized):
    lm = normalized.reshape(len(normalized), 33, 4)
    knee_y = (lm[:, 25, 1] + lm[:, 26, 1]) / 2
    return knee_y.min()

def body_measurements(sequence):
    lm = sequence.reshape(len(sequence), 33, 4)
    x = lm[..., 0] * aspect
    y = lm[..., 1]
    hip_x = (x[:, 23] + x[:, 24]) / 2
    hip_y = (y[:, 23] + y[:, 24]) / 2
    sh_x = (x[:, 11] + x[:, 12]) / 2
    sh_y = (y[:, 11] + y[:, 12]) / 2
    torso = np.hypot(sh_x - hip_x, sh_y - hip_y).max() + 1e-6
    caves = []
    for hip, knee, ankle in [(23, 25, 27), (24, 26, 28)]:
        t = (y[:, knee] - y[:, hip]) / (y[:, ankle] - y[:, hip] + 1e-9)
        line_x = x[:, hip] + t * (x[:, ankle] - x[:, hip])
        caves.append((np.abs(line_x - hip_x) - np.abs(x[:, knee] - hip_x)) / torso)
    knee_y = (y[:, 25] + y[:, 26]) / 2
    ankle_y = (y[:, 27] + y[:, 28]) / 2
    return np.stack([(knee_y - hip_y) / torso, (ankle_y - hip_y) / torso,
                    caves[0], caves[1],
                    np.abs(x[:, 25] - x[:, 26]) / torso,
                    np.abs(x[:, 27] - x[:, 28]) / torso,], axis=1).astype(np.float32)
