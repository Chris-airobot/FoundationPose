import os, sys, base64, time, csv, json, math
from pathlib import Path

import cv2, zmq, msgpack, numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from estimater import *

ENDPOINT = 'tcp://192.168.123.164:5555'
INIT = ROOT / 'g1/data/live_init'
OUT = ROOT / 'g1/results/range_benchmark'
K = np.loadtxt(INIT / 'cam_K.txt').reshape(3, 3)

TRACK_ITERS = 1
REGISTER_ITERS = 5
TRACK_FRAMES = 60
VIS_EVERY = 5

# Stationary-object heuristic only; this is not ground-truth pose accuracy.
STABLE_STEP_TRANSLATION_MM = 20.0
STABLE_STEP_ROTATION_DEG = 10.0


def decode(payload):
    data = msgpack.unpackb(payload, raw=False)
    out = {}
    for k, v in data['images'].items():
        b = base64.b64decode(v) if isinstance(v, str) else v
        out[k] = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    return out


def depth_m(depth_u16):
    d = depth_u16.astype(np.float32)
    d *= 0.001
    d[(d < 0.001) | (d > 10.0)] = 0
    return d


def recv_rgbd(sock):
    while True:
        im = decode(sock.recv())
        if 'ego_view' in im and 'ego_view_depth' in im:
            return im['ego_view'], im['ego_view_depth']


def mask_ui(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    shown = bgr.copy()
    pts = []
    win = 'Range benchmark: outline visible box'

    def mouse(event, x, y, flags, param):
        nonlocal shown
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            shown = bgr.copy()
            if len(pts) > 1:
                cv2.polylines(shown, [np.asarray(pts)], False, (0, 255, 0), 2)
            for p in pts:
                cv2.circle(shown, p, 4, (0, 0, 255), -1)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse)
    print('Mask: left-click visible box; ENTER accept; R reset; ESC cancel')
    while True:
        frame = shown.copy()
        cv2.putText(frame, 'ENTER accept | R reset | ESC cancel',
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(pts) >= 3:
            mask = np.zeros(rgb.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.asarray(pts)], 1)
            cv2.destroyWindow(win)
            return mask.astype(bool)
        if key == ord('r'):
            pts.clear()
            shown = bgr.copy()
        if key == 27:
            cv2.destroyWindow(win)
            return None


def depth_stats(depth, mask):
    total = int(mask.sum())
    valid = mask & np.isfinite(depth) & (depth > 0)
    nvalid = int(valid.sum())
    if total == 0 or nvalid == 0:
        return dict(distance_m=float('nan'), valid_depth_pct=0.0,
                    depth_p10_m=float('nan'), depth_p90_m=float('nan'))
    vals = depth[valid]
    return dict(
        distance_m=float(np.median(vals)),
        valid_depth_pct=100.0 * nvalid / total,
        depth_p10_m=float(np.percentile(vals, 10)),
        depth_p90_m=float(np.percentile(vals, 90)),
    )


def rotation_angle_deg(Ra, Rb):
    R = Ra.T @ Rb
    c = (np.trace(R) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


def pose_metrics(poses):
    if len(poses) < 2:
        return dict(translation_spread_p95_mm=float('nan'),
                    rotation_spread_p95_deg=float('nan'),
                    step_translation_p95_mm=float('nan'),
                    step_rotation_p95_deg=float('nan'),
                    stable_heuristic=False)

    ts = np.asarray([p[:3, 3] for p in poses])
    t_ref = np.median(ts, axis=0)
    spread_t = np.linalg.norm(ts - t_ref[None, :], axis=1) * 1000.0
    R_ref = poses[0][:3, :3]
    spread_r = np.asarray([rotation_angle_deg(R_ref, p[:3, :3]) for p in poses])
    step_t = np.linalg.norm(np.diff(ts, axis=0), axis=1) * 1000.0
    step_r = np.asarray([
        rotation_angle_deg(poses[i - 1][:3, :3], poses[i][:3, :3])
        for i in range(1, len(poses))
    ])
    step_t_p95 = float(np.percentile(step_t, 95))
    step_r_p95 = float(np.percentile(step_r, 95))
    stable = (step_t_p95 <= STABLE_STEP_TRANSLATION_MM and
              step_r_p95 <= STABLE_STEP_ROTATION_DEG)
    return dict(
        translation_spread_p95_mm=float(np.percentile(spread_t, 95)),
        rotation_spread_p95_deg=float(np.percentile(spread_r, 95)),
        step_translation_p95_mm=step_t_p95,
        step_rotation_p95_deg=step_r_p95,
        stable_heuristic=bool(stable),
    )


def append_csv(path, row):
    write_header = not path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    set_logging_format()
    set_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(ROOT / 'box.obj')
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    center_tf = np.linalg.inv(to_origin)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
        debug_dir=str(OUT), debug=0, glctx=dr.RasterizeCudaContext())

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, '')
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(ENDPOINT)

    csv_path = OUT / 'range_results.csv'
    sample_id = 0
    print('RANGE BENCHMARK')
    print('Move box roughly closer/farther. S tests current position; Q quits.')
    print(f'Each sample registers once then tracks {TRACK_FRAMES} live RGB-D frames.')

    try:
        while True:
            rgb, depth_raw = recv_rgbd(sock)
            preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(preview, 'S: sample current position | Q: quit',
                        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow('FoundationPose range benchmark', preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key != ord('s'):
                continue

            depth = depth_m(depth_raw)
            mask = mask_ui(rgb)
            if mask is None:
                print('Sample cancelled.')
                continue

            stats = depth_stats(depth, mask)
            if not np.isfinite(stats['distance_m']):
                print('No valid depth inside mask; sample not run.')
                continue

            sample_id += 1
            sample_dir = OUT / f'sample_{sample_id:03d}'
            sample_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sample_dir / 'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(sample_dir / 'depth.png'), depth_raw)
            cv2.imwrite(str(sample_dir / 'mask.png'), mask.astype(np.uint8) * 255)

            print(f'SAMPLE {sample_id}: distance={stats["distance_m"]:.3f} m | '
                  f'valid depth={stats["valid_depth_pct"]:.1f}%')

            register_ok = False
            error = ''
            register_ms = float('nan')
            poses, track_ms = [], []
            elapsed = float('nan')

            try:
                t0 = time.perf_counter()
                pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                    iteration=REGISTER_ITERS)
                register_ms = (time.perf_counter() - t0) * 1000.0
                if not np.isfinite(pose).all():
                    raise RuntimeError('registration returned non-finite pose')
                register_ok = True
                np.savetxt(sample_dir / 'register_pose.txt', pose)

                track_start = time.perf_counter()
                for j in range(TRACK_FRAMES):
                    rgb_t, depth_raw_t = recv_rgbd(sock)
                    depth_t = depth_m(depth_raw_t)
                    t1 = time.perf_counter()
                    pose = est.track_one(rgb=rgb_t, depth=depth_t, K=K,
                                         iteration=TRACK_ITERS)
                    track_ms.append((time.perf_counter() - t1) * 1000.0)
                    poses.append(pose.copy())

                    if j % VIS_EVERY == 0:
                        cp = pose @ center_tf
                        vis = draw_posed_3d_box(K, img=rgb_t, ob_in_cam=cp, bbox=bbox)
                        vis = draw_xyz_axis(vis, ob_in_cam=cp, scale=0.1, K=K,
                                            thickness=3, transparency=0,
                                            is_input_rgb=True)
                        show = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                        cv2.putText(show,
                                    f'sample {sample_id} | {stats["distance_m"]:.2f} m | {j+1}/{TRACK_FRAMES}',
                                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0), 2)
                        cv2.imshow('FoundationPose range benchmark', show)
                        cv2.waitKey(1)

                elapsed = time.perf_counter() - track_start
                np.savetxt(sample_dir / 'latest_pose.txt', poses[-1])

            except Exception as e:
                error = f'{type(e).__name__}: {e}'
                print('Sample error:', error)

            pm = pose_metrics(poses)
            end_to_end_fps = len(poses) / elapsed if poses and elapsed > 0 else 0.0
            row = {
                'sample': sample_id,
                'timestamp': time.time(),
                'distance_m': stats['distance_m'],
                'valid_depth_pct': stats['valid_depth_pct'],
                'depth_p10_m': stats['depth_p10_m'],
                'depth_p90_m': stats['depth_p90_m'],
                'register_ok': register_ok,
                'register_ms': register_ms,
                'tracked_frames': len(poses),
                'end_to_end_pose_fps': end_to_end_fps,
                'mean_track_ms': float(np.mean(track_ms)) if track_ms else float('nan'),
                'p95_track_ms': float(np.percentile(track_ms, 95)) if track_ms else float('nan'),
                **pm,
                'error': error,
            }
            append_csv(csv_path, row)
            (sample_dir / 'metrics.json').write_text(json.dumps(row, indent=2))

            print(f'RESULT {sample_id}: {stats["distance_m"]:.3f} m | '
                  f'depth={stats["valid_depth_pct"]:.1f}% | '
                  f'pose FPS={end_to_end_fps:.1f} | '
                  f'step p95={pm["step_translation_p95_mm"]:.1f} mm / '
                  f'{pm["step_rotation_p95_deg"]:.1f} deg | '
                  f'stable={pm["stable_heuristic"]}')
            print('Move box to another rough distance and press S again.')

    finally:
        cv2.destroyAllWindows()
        sock.close(0)
        ctx.term()
        print('Saved:', csv_path)


if __name__ == '__main__':
    main()
