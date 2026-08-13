import os, sys, time, csv, json, base64, argparse, shutil
from pathlib import Path

import cv2
import zmq
import msgpack
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENDPOINT = 'tcp://192.168.123.164:5555'
DEFAULT_K = ROOT / 'g1/data/live_init/cam_K.txt'
APRIL_CFG = ROOT / 'g1/config/apriltag_gt_placeholder.json'


def decode(payload):
    data = msgpack.unpackb(payload, raw=False)
    out = {}
    for k, v in data['images'].items():
        b = base64.b64decode(v) if isinstance(v, str) else v
        out[k] = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    return out


def recv_rgbd(sock):
    while True:
        im = decode(sock.recv())
        if 'ego_view' in im and 'ego_view_depth' in im:
            return im['ego_view'], im['ego_view_depth']


def parse_args():
    p = argparse.ArgumentParser(description='Record a rosbag-like offline RGB-D bundle from the G1 camera stream.')
    p.add_argument('--endpoint', default=DEFAULT_ENDPOINT)
    p.add_argument('--seconds', type=float, default=60.0,
                   help='Recording duration. Use <=0 to record until Ctrl+C.')
    p.add_argument('--out', default='',
                   help='Output folder. Default: g1/data/offline_bundle_<timestamp>')
    p.add_argument('--every', type=int, default=1,
                   help='Save every Nth received RGB-D frame (default 1).')
    p.add_argument('--robot-marker-tf', default='',
                   help='Optional 4x4 txt file for one independently measured T_robot_marker reference pose.')
    return p.parse_args()


def main():
    args = parse_args()
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = Path(args.out) if args.out else ROOT / f'g1/data/offline_bundle_{stamp}'
    rgb_dir = out / 'rgb'
    depth_dir = out / 'depth'
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    K = np.loadtxt(DEFAULT_K).reshape(3, 3)
    np.savetxt(out / 'cam_K.txt', K)
    if APRIL_CFG.exists():
        shutil.copy2(APRIL_CFG, out / 'apriltag_config.json')
    if (ROOT / 'box.obj').exists():
        shutil.copy2(ROOT / 'box.obj', out / 'box.obj')

    if args.robot_marker_tf:
        src = Path(args.robot_marker_tf)
        T = np.loadtxt(src).reshape(4, 4)
        np.savetxt(out / 'T_robot_marker_reference.txt', T)

    metadata = {
        'format': 'g1_offline_rgbd_bundle_v1',
        'created_unix_s': time.time(),
        'endpoint': args.endpoint,
        'rgb_key': 'ego_view',
        'depth_key': 'ego_view_depth',
        'depth_unit': 'uint16_mm',
        'camera_intrinsics_file': 'cam_K.txt',
        'apriltag_config_file': 'apriltag_config.json' if APRIL_CFG.exists() else None,
        'box_mesh_file': 'box.obj' if (ROOT / 'box.obj').exists() else None,
        'robot_stationary_during_capture': True,
        'T_robot_camera': None,
        'T_robot_marker_reference_file': 'T_robot_marker_reference.txt' if args.robot_marker_tf else None,
        'note': 'Raw RGB-D is sufficient to rerun FoundationPose and AprilTag offline. For robot-frame poses later, add one verified static T_robot_camera; alternatively derive it from one independent T_robot_marker reference and the simultaneous vision T_camera_marker.'
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, '')
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(args.endpoint)

    timestamps_path = out / 'timestamps.csv'
    f = timestamps_path.open('w', newline='')
    writer = csv.DictWriter(f, fieldnames=['frame', 'unix_s', 'relative_s', 'rgb_file', 'depth_file'])
    writer.writeheader()

    print('OFFLINE RGB-D BUNDLE RECORDER')
    print('output:', out)
    print('endpoint:', args.endpoint)
    print('duration:', 'until Ctrl+C' if args.seconds <= 0 else f'{args.seconds:.1f} s')
    print('robot assumed stationary; raw RGB-D + K + AprilTag config are being saved')

    t0 = time.perf_counter()
    received = 0
    saved = 0
    try:
        while True:
            rgb, depth = recv_rgbd(sock)
            received += 1
            if received % max(args.every, 1) != 0:
                continue

            rel = time.perf_counter() - t0
            idx = saved
            rgb_name = f'{idx:06d}.png'
            depth_name = f'{idx:06d}.png'

            # Stream RGB is RGB; cv2.imwrite expects BGR for visually correct PNG.
            cv2.imwrite(str(rgb_dir / rgb_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(depth_dir / depth_name), depth)
            writer.writerow({
                'frame': idx,
                'unix_s': f'{time.time():.9f}',
                'relative_s': f'{rel:.9f}',
                'rgb_file': f'rgb/{rgb_name}',
                'depth_file': f'depth/{depth_name}',
            })
            saved += 1

            if saved % 30 == 0:
                print(f'saved={saved} | elapsed={rel:.1f}s | approx_save_fps={saved/max(rel,1e-6):.1f}')

            if args.seconds > 0 and rel >= args.seconds:
                break
    except KeyboardInterrupt:
        print('\nStopped by Ctrl+C.')
    finally:
        f.flush()
        f.close()
        sock.close(0)
        ctx.term()

    metadata['frames_saved'] = saved
    metadata['recording_seconds'] = time.perf_counter() - t0
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    print('DONE')
    print('frames:', saved)
    print('bundle:', out)
    print('next: python g1/scripts/verify_offline_bundle.py --bundle', out)


if __name__ == '__main__':
    main()
