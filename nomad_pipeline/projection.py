import numpy as np


def project_points(points_xy, K, h_cam=0.15, min_depth=0.1, cam_dx=0.0, cam_dy=0.0,
                   img_size=None, pitch=0.0):
    """Проєкція точок (x вперед, y ліворуч, z відносна висота) на кадр.

    Xc = -y, Yc = h_cam, Zc = x
    Для дрона в польоті h_cam ~ 0.0..0.2м (на рівні польоту або трохи нижче горизонту камери).
    """
    p = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    x = p[:, 0] - cam_dx
    y = p[:, 1] - cam_dy
    Xc, Yc, Zc = -y, np.full_like(x, h_cam), x
    if pitch:
        c, s = np.cos(pitch), np.sin(pitch)
        Yc, Zc = Yc * c - Zc * s, Yc * s + Zc * c
    valid = Zc >= min_depth
    Zs = np.where(valid, Zc, 1.0)
    u = K[0, 0] * Xc / Zs + K[0, 2]
    v = K[1, 1] * Yc / Zs + K[1, 2]
    uv = np.stack([u, v], axis=1)
    if img_size is not None:
        w, h = img_size
        valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return uv, valid


def project_corridor(points_xy, K, half_width=0.20, h_cam=0.15, min_depth=0.1, pitch=0.0):
    """Генерує ліву та праву межі польотного коридору дрона в повітрі."""
    p = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        return None, None
    left_pts = p.copy()
    left_pts[:, 1] += half_width  # y ліворуч
    right_pts = p.copy()
    right_pts[:, 1] -= half_width # y праворуч
    uv_l, ok_l = project_points(left_pts, K, h_cam=h_cam, min_depth=min_depth, pitch=pitch)
    uv_r, ok_r = project_points(right_pts, K, h_cam=h_cam, min_depth=min_depth, pitch=pitch)
    return (uv_l, ok_l), (uv_r, ok_r)
