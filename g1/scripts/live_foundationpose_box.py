import os, sys, base64
from pathlib import Path
import cv2, zmq, msgpack, numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from estimater import *

ENDPOINT='tcp://192.168.123.164:5555'
INIT=ROOT/'g1/data/live_init'
OUT=ROOT/'g1/results/live_foundationpose'
K=np.loadtxt(INIT/'cam_K.txt').reshape(3,3)


def decode(payload):
    data=msgpack.unpackb(payload,raw=False)
    out={}
    for k,v in data['images'].items():
        b=base64.b64decode(v) if isinstance(v,str) else v
        out[k]=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED)
    return out


def depth_m(d):
    d=d.astype(np.float32)/1000.0
    d[(d<0.001)|(d>10.0)]=0
    return d


def mask_ui(rgb):
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR); shown=bgr.copy(); pts=[]
    win='Reinitialize mask'
    def mouse(e,x,y,f,p):
        nonlocal shown
        if e==cv2.EVENT_LBUTTONDOWN:
            pts.append((x,y)); shown=bgr.copy()
            if len(pts)>1: cv2.polylines(shown,[np.array(pts)],False,(0,255,0),2)
            for q in pts: cv2.circle(shown,q,4,(0,0,255),-1)
    cv2.namedWindow(win); cv2.setMouseCallback(win,mouse)
    print('Left click box; ENTER accept; R reset; ESC cancel')
    while True:
        cv2.imshow(win,shown); key=cv2.waitKey(20)&0xFF
        if key==13 and len(pts)>=3:
            m=np.zeros(rgb.shape[:2],np.uint8); cv2.fillPoly(m,[np.array(pts)],1)
            cv2.destroyWindow(win); return m.astype(bool)
        if key==ord('r'): pts.clear(); shown=bgr.copy()
        if key==27: cv2.destroyWindow(win); return None


def main():
    set_logging_format(); set_seed(0); os.makedirs(OUT,exist_ok=True)
    mesh=trimesh.load(ROOT/'box.obj')
    to_origin,extents=trimesh.bounds.oriented_bounds(mesh)
    bbox=np.stack([-extents/2,extents/2],axis=0).reshape(2,3)
    est=FoundationPose(model_pts=mesh.vertices,model_normals=mesh.vertex_normals,mesh=mesh,
        scorer=ScorePredictor(),refiner=PoseRefinePredictor(),debug_dir=str(OUT),debug=1,
        glctx=dr.RasterizeCudaContext())

    bgr=cv2.imread(str(INIT/'rgb/000000.png'))
    rgb0=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    dep0=depth_m(cv2.imread(str(INIT/'depth/000000.png'),-1))
    mask=cv2.imread(str(INIT/'masks/000000.png'),-1).astype(bool)
    pose=est.register(K=K,rgb=rgb0,depth=dep0,ob_mask=mask,iteration=5)
    np.savetxt(OUT/'init_pose.txt',pose)

    ctx=zmq.Context(); sock=ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE,''); sock.setsockopt(zmq.CONFLATE,1)
    sock.connect(ENDPOINT)
    print('LIVE: q quit, r redraw mask/re-register')
    i=0
    try:
        while True:
            im=decode(sock.recv())
            if 'ego_view' not in im or 'ego_view_depth' not in im: continue
            rgb=im['ego_view']; dep=depth_m(im['ego_view_depth'])
            pose=est.track_one(rgb=rgb,depth=dep,K=K,iteration=2)
            np.savetxt(OUT/'latest_pose.txt',pose)
            cp=pose@np.linalg.inv(to_origin)
            vis=draw_posed_3d_box(K,img=rgb,ob_in_cam=cp,bbox=bbox)
            vis=draw_xyz_axis(vis,ob_in_cam=cp,scale=0.1,K=K,thickness=3,transparency=0,is_input_rgb=True)
            show=vis[...,::-1].copy()
            cv2.putText(show,f'frame {i} | q quit | r reinit',(15,28),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,0),2)
            cv2.imshow('G1 FoundationPose Live',show)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'): break
            if key==ord('r'):
                m=mask_ui(rgb)
                if m is not None:
                    pose=est.register(K=K,rgb=rgb,depth=dep,ob_mask=m,iteration=5)
            i+=1
    finally:
        cv2.destroyAllWindows(); sock.close(0); ctx.term()

if __name__=='__main__': main()
