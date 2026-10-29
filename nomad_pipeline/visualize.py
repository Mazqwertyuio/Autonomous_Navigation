import cv2
import numpy as np
from .projection import project_points, project_corridor


def draw_trajectory(frame, waypoints_xy, K, cfg, vw=None, fps=None, color=(0, 255, 0),
                    is_primary=True, mode_str=None, dist=None, shield_info=None):
    """Малює 3D AR польотний коридор (стрічку) в повітрі для дрона."""
    out = frame.copy()
    h, w = out.shape[:2]
    pts = np.asarray(waypoints_xy).reshape(-1, 2)
    if len(pts) == 0:
        return out

    # Додаємо вихідну точку перед дроном
    pts_full = np.vstack([[0.25, 0.0], pts])

    # Колір стрічки залежно від стану щита безпеки
    ribbon_fill = (0, 180, 80)
    ribbon_edge = (0, 255, 120)
    if shield_info is not None:
        st = shield_info.get("state", "CLEAR_FORWARD")
        if st == "EMERGENCY_BRAKE":
            ribbon_fill = (0, 30, 210)
            ribbon_edge = (0, 60, 255)
        elif st == "CAUTION_SLOW":
            ribbon_fill = (0, 165, 230)
            ribbon_edge = (0, 215, 255)

    # 1. Якщо це головна обрана траєкторія - малюємо 3D AR Flight Ribbon
    if is_primary and len(pts_full) >= 3:
        (uv_l, ok_l), (uv_r, ok_r) = project_corridor(
            pts_full, K, half_width=0.18, h_cam=cfg.h_cam, pitch=np.radians(cfg.pitch_deg)
        )
        valid_pairs = []
        for (pl, okl), (pr, okr) in zip(zip(uv_l, ok_l), zip(uv_r, ok_r)):
            if okl and okr and (-w < pl[0] < 2 * w) and (-w < pr[0] < 2 * h):
                valid_pairs.append(((int(pl[0]), int(pl[1])), (int(pr[0]), int(pr[1]))))
        if len(valid_pairs) >= 2:
            left_edge = [p[0] for p in valid_pairs]
            right_edge = [p[1] for p in valid_pairs]
            poly = np.array(left_edge + right_edge[::-1], dtype=np.int32)

            overlay = out.copy()
            cv2.fillPoly(overlay, [poly], ribbon_fill)
            cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

            for i in range(len(left_edge) - 1):
                cv2.line(out, left_edge[i], left_edge[i + 1], ribbon_edge, 2, cv2.LINE_AA)
                cv2.line(out, right_edge[i], right_edge[i + 1], ribbon_edge, 2, cv2.LINE_AA)

    # 2. Центральна лінія траєкторії та точки-бусини (waypoint beads)
    uv, valid = project_points(pts_full, K, h_cam=cfg.h_cam, min_depth=cfg.min_depth,
                               pitch=np.radians(cfg.pitch_deg))
    prev = None
    for (u, v), ok in zip(uv, valid):
        if not ok or not (-w < u < 2 * w and -h < v < 2 * h):
            prev = None
            continue
        c = (int(u), int(v))
        if prev is not None:
            cv2.line(out, prev, c, color if not is_primary else (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, c, 5 if is_primary else 3, (0, 0, 255) if not is_primary else ribbon_edge, -1, cv2.LINE_AA)
        if is_primary:
            cv2.circle(out, c, 2, (255, 255, 255), -1, cv2.LINE_AA)
        prev = c

    # 3. HUD інформація польоту
    txt = []
    if vw:
        txt.append(f"Fwd={vw[0]:.2f}m/s  Yaw={vw[1]:.2f}rad/s")
    if dist is not None:
        txt.append(f"Dist={dist:.1f}")
    if fps:
        txt.append(f"FPS={fps:.1f}")
    if mode_str:
        txt.append(f"[{mode_str}]")
    if txt:
        cv2.rectangle(out, (5, 5), (575, 36), (0, 0, 0), -1)
        cv2.putText(out, "  ".join(txt), (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    # 4. Індикатор тактичного щита безпеки (якщо активний)
    if shield_info is not None and is_primary:
        st = shield_info["state"]
        sc = int(shield_info["speed_scale"] * 100)
        badge_col = (0, 200, 0) if st == "CLEAR_FORWARD" else ((0, 180, 255) if st == "CAUTION_SLOW" else (0, 0, 255))
        if "coll_prob" in shield_info:
            detail = f"Pcoll={shield_info['coll_prob']:.0%}"
        else:
            detail = f"{shield_info['min_dist_m']:.2f}m"
        label = f"SHIELD: {st} ({detail} | speed {sc}%)"
        cv2.rectangle(out, (5, 40), (390, 68), (15, 15, 15), -1)
        cv2.putText(out, label, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, badge_col, 2, cv2.LINE_AA)

    return out
