#!/usr/bin/env python3
"""Web-based Ropedia episode viewer.

Displays RGB images, colorized depth (with confidence filtering), and a 3D
point cloud side-by-side.  Browse scenes/episodes from the dataset root and
play through frames with adjustable FPS.

Usage:
    python scripts/visualize_ropedia_web.py
    python scripts/visualize_ropedia_web.py --data-dir datasets/ropedia/datasets --port 8080
"""

import argparse
import io
import struct
from functools import lru_cache
from pathlib import Path

import cv2
import h5py
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
import uvicorn

# ── Global ──
BASE_DIR: Path = Path("datasets/ropedia/datasets")
app = FastAPI(title="Ropedia Viewer")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _episode_meta(ep_dir_str: str):
    ep_dir = Path(ep_dir_str)
    pose_data = np.load(str(ep_dir / "pose" / "left.npz"))
    poses = pose_data["data"]          # (N, 4, 4) T_world_camera
    pose_inds = pose_data["inds"]      # (N,)

    with h5py.File(str(ep_dir / "annotation.hdf5"), "r") as f:
        K = f["calibration/cam01/K"][:].tolist()  # [fx, fy, cx, cy]

    n_images = len(list((ep_dir / "images" / "left").glob("frame_*_rgb.png")))
    n_frames = min(len(pose_inds), n_images)
    return {
        "poses": poses,
        "pose_inds": pose_inds,
        "K": K,
        "n_frames": n_frames,
    }


def _ep_dir(scene_id: str, ep_name: str) -> Path:
    return BASE_DIR / scene_id / ep_name


def _frame_name(meta: dict, idx: int) -> str:
    return f"frame_{int(meta['pose_inds'][idx]):05d}_rgb.png"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/scenes")
def list_scenes():
    return sorted(d.name for d in BASE_DIR.iterdir() if d.is_dir())


@app.get("/api/episodes/{scene_id}")
def list_episodes(scene_id: str):
    scene_dir = BASE_DIR / scene_id
    return sorted(
        d.name
        for d in scene_dir.iterdir()
        if d.is_dir() and (d / "annotation.hdf5").exists()
    )


@app.get("/api/meta/{scene_id}/{ep_name}")
def get_meta(scene_id: str, ep_name: str):
    meta = _episode_meta(str(_ep_dir(scene_id, ep_name)))
    return {"n_frames": meta["n_frames"], "K": meta["K"]}


@app.get("/api/rgb/{scene_id}/{ep_name}/{idx}")
def get_rgb(scene_id: str, ep_name: str, idx: int):
    ep = _ep_dir(scene_id, ep_name)
    meta = _episode_meta(str(ep))
    img = cv2.imread(str(ep / "images" / "left" / _frame_name(meta, idx)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/depth/{scene_id}/{ep_name}/{idx}")
def get_depth_viz(
    scene_id: str,
    ep_name: str,
    idx: int,
    conf: float = Query(0.3),
    max_depth: float = Query(10.0),
):
    ep = _ep_dir(scene_id, ep_name)
    meta = _episode_meta(str(ep))
    fname = _frame_name(meta, idx)

    depth = cv2.imread(str(ep / "depths" / fname), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    conf_map = cv2.imread(str(ep / "conf_mask" / fname), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0

    valid = (depth > 0.01) & (depth < max_depth) & (conf_map > conf)

    depth_norm = np.zeros_like(depth)
    if valid.any():
        depth_norm[valid] = np.clip(depth[valid] / max_depth, 0, 1)

    depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_colored[~valid] = 0

    _, buf = cv2.imencode(".jpg", depth_colored, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/pcd/{scene_id}/{ep_name}/{idx}")
def get_pcd(
    scene_id: str,
    ep_name: str,
    idx: int,
    downsample: int = Query(4),
    conf: float = Query(0.3),
    max_depth: float = Query(10.0),
):
    """Return binary: int32(N) | float32[16](pose) | float32[4](K) | float32[N*3](xyz) | uint8[N*3](rgb)."""
    ep = _ep_dir(scene_id, ep_name)
    meta = _episode_meta(str(ep))
    fname = _frame_name(meta, idx)
    K = np.array(meta["K"], dtype=np.float32)
    fx, fy, cx, cy = K

    img = cv2.cvtColor(cv2.imread(str(ep / "images" / "left" / fname)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(ep / "depths" / fname), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    conf_map = cv2.imread(str(ep / "conf_mask" / fname), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0

    H, W = depth.shape
    v_idx = np.arange(0, H, downsample)
    u_idx = np.arange(0, W, downsample)
    uu, vv = np.meshgrid(u_idx, v_idx)

    d = depth[vv, uu]
    c = conf_map[vv, uu]
    rgb = img[vv, uu]
    valid = (d > 0.01) & (d < max_depth) & (c > conf)

    X = (uu[valid].astype(np.float32) - cx) / fx * d[valid]
    Y = (vv[valid].astype(np.float32) - cy) / fy * d[valid]
    Z = d[valid]
    pts_cam = np.stack([X, Y, Z], axis=-1)
    colors = rgb[valid]

    pose = meta["poses"][idx].astype(np.float32)
    R, t = pose[:3, :3], pose[:3, 3]
    pts_world = (pts_cam @ R.T + t).astype(np.float32)

    buf = io.BytesIO()
    buf.write(struct.pack("<i", len(pts_world)))
    buf.write(pose.tobytes())
    buf.write(K.tobytes())
    buf.write(pts_world.tobytes())
    buf.write(colors.astype(np.uint8).tobytes())
    return Response(content=buf.getvalue(), media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ropedia Viewer</title>
<style>
:root{
  --bg:#0d1117; --sf:#161b22; --sf2:#21262d; --bd:#30363d;
  --tx:#e6edf3; --txm:#8b949e; --pr:#58a6ff; --pr2:#1f6feb;
  --rd:#f85149; --gn:#3fb950;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--tx);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;overflow:hidden;height:100vh;display:flex;flex-direction:column}

/* ─── header ─── */
.hdr{display:flex;align-items:center;gap:14px;padding:6px 14px;background:var(--sf);border-bottom:1px solid var(--bd);flex-wrap:wrap}
.hdr label{color:var(--txm);font-size:10px;text-transform:uppercase;letter-spacing:.6px;display:flex;flex-direction:column;gap:2px}
.hdr select{background:var(--sf2);color:var(--tx);border:1px solid var(--bd);padding:3px 6px;border-radius:4px;font-size:12px;max-width:260px}
.pb{display:flex;align-items:center;gap:4px}
.pb button{background:var(--sf2);color:var(--tx);border:1px solid var(--bd);width:30px;height:28px;border-radius:4px;cursor:pointer;font-size:13px;display:flex;align-items:center;justify-content:center}
.pb button:hover{background:var(--pr2);border-color:var(--pr)}
.sr{display:flex;align-items:center;gap:5px}
.sr input[type=range]{width:110px;accent-color:var(--pr);height:4px}
.sr .v{color:var(--pr);min-width:56px;text-align:right;font-variant-numeric:tabular-nums;font-size:12px}
.badge{background:var(--sf2);color:var(--gn);padding:2px 8px;border-radius:10px;font-size:11px;font-variant-numeric:tabular-nums}

/* ─── main ─── */
.main{flex:1;display:flex;overflow:hidden}

/* left */
.lp{width:32%;min-width:260px;display:flex;flex-direction:column;border-right:1px solid var(--bd);overflow:hidden}
.img-sec{flex:1;min-height:0;padding:4px 6px;display:flex;flex-direction:column}
.img-lbl{font-size:10px;color:var(--txm);text-transform:uppercase;letter-spacing:.6px;margin-bottom:2px;flex-shrink:0}
.img-sec img{width:100%;flex:1;min-height:0;object-fit:contain;border-radius:4px;background:var(--sf2);display:block}
.ctrl{flex-shrink:0;padding:5px 10px;background:var(--sf);border-top:1px solid var(--bd)}
.cr{display:flex;align-items:center;gap:6px;margin:2px 0}
.cr label{flex:0 0 100px;color:var(--txm);font-size:11px}
.cr input[type=range]{flex:1;accent-color:var(--pr);height:4px}
.cr .v{color:var(--pr);min-width:44px;text-align:right;font-variant-numeric:tabular-nums;font-size:12px}

/* right */
.rp{flex:1;position:relative;display:flex;flex-direction:column}
.rp canvas{flex:1;display:block}
.rp-ctrl{position:absolute;bottom:8px;right:8px;background:rgba(22,27,34,.92);padding:6px 10px;border-radius:6px;border:1px solid var(--bd);backdrop-filter:blur(6px)}
.rp-info{position:absolute;top:8px;left:8px;background:rgba(22,27,34,.82);padding:4px 10px;border-radius:6px;font-size:11px;color:var(--txm);font-variant-numeric:tabular-nums}
</style>
</head>
<body>

<!-- header -->
<div class="hdr">
  <label>Scene<select id="sc"></select></label>
  <label>Episode<select id="ep"></select></label>
  <div class="pb">
    <button id="prev" title="Prev">⏮</button>
    <button id="play" title="Play / Pause">▶</button>
    <button id="next" title="Next">⏭</button>
  </div>
  <div class="sr">
    <span style="color:var(--txm);font-size:11px">Frame</span>
    <input type="range" id="frm" min="0" max="0" value="0">
    <span class="v" id="frm-v">0 / 0</span>
  </div>
  <div class="sr">
    <span style="color:var(--txm);font-size:11px">FPS</span>
    <input type="range" id="fps" min="0.5" max="30" step="0.5" value="1">
    <span class="v" id="fps-v">1.0</span>
  </div>
  <span class="badge" id="status">Ready</span>
</div>

<!-- main -->
<div class="main">
  <!-- left: images -->
  <div class="lp">
    <div class="img-sec"><div class="img-lbl">RGB</div><img id="img-rgb" alt="RGB"></div>
    <div class="img-sec"><div class="img-lbl">Depth (colorized)</div><img id="img-dep" alt="Depth"></div>
    <div class="ctrl">
      <div class="cr"><label>Conf Threshold</label><input type="range" id="conf" min="0" max="1" step="0.01" value="0.3"><span class="v" id="conf-v">0.30</span></div>
      <div class="cr"><label>Max Depth (m)</label><input type="range" id="maxd" min="0.5" max="50" step="0.5" value="10"><span class="v" id="maxd-v">10.0</span></div>
    </div>
  </div>

  <!-- right: 3D -->
  <div class="rp">
    <canvas id="cv"></canvas>
    <div class="rp-info" id="pcd-info">0 pts</div>
    <div class="rp-ctrl">
      <div class="cr"><label>Downsample</label><input type="range" id="ds" min="1" max="32" step="1" value="4"><span class="v" id="ds-v">4</span></div>
      <div class="cr"><label>Point Size</label><input type="range" id="ps" min="1" max="80" step="1" value="10"><span class="v" id="ps-v">0.010</span></div>
    </div>
  </div>
</div>

<script type="importmap">{"imports":{"three":"https://esm.sh/three@0.169.0","three/addons/":"https://esm.sh/three@0.169.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

/* ── helpers ── */
const $=id=>document.getElementById(id);
const json=u=>fetch(u).then(r=>r.json());

/* ── state ── */
const S={sc:null,ep:null,n:0,K:[],f:0,playing:false,fps:1,conf:.3,maxd:10,ds:4,ps:.01};
let loadSeq=0;

/* ── DOM refs ── */
const scSel=$('sc'),epSel=$('ep');
const prevBtn=$('prev'),playBtn=$('play'),nextBtn=$('next');
const frmR=$('frm'),frmV=$('frm-v');
const fpsR=$('fps'),fpsV=$('fps-v');
const confR=$('conf'),confV=$('conf-v');
const maxdR=$('maxd'),maxdV=$('maxd-v');
const dsR=$('ds'),dsV=$('ds-v');
const psR=$('ps'),psV=$('ps-v');
const rgbImg=$('img-rgb'),depImg=$('img-dep');
const canvas=$('cv');
const statusBadge=$('status');
const pcdInfo=$('pcd-info');

/* ── Three.js setup ── */
const scene=new THREE.Scene();
scene.background=new THREE.Color(0x0d1117);
const cam=new THREE.PerspectiveCamera(60,1,.01,500);
cam.position.set(0,-1.5,-2.5);
cam.up.set(0,-1,0);
const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
const ctrl=new OrbitControls(cam,canvas);
ctrl.enableDamping=true;ctrl.dampingFactor=.12;

scene.add(new THREE.AxesHelper(.3));

const ambLight=new THREE.AmbientLight(0xffffff,.6);
scene.add(ambLight);

/* point cloud */
const pGeo=new THREE.BufferGeometry();
const pMat=new THREE.PointsMaterial({size:S.ps,vertexColors:true,sizeAttenuation:true});
const pObj=new THREE.Points(pGeo,pMat);
scene.add(pObj);

/* camera frustum lines */
let frustumObj=null;
function drawFrustum(pose,K){
  if(frustumObj){scene.remove(frustumObj);frustumObj.geometry.dispose();}
  const [fx,fy,cx,cy]=K, w=512,h=512,sc=.12;
  const corners=[[0,0],[w,0],[w,h],[0,h]].map(([u,v])=>[
    (u-cx)/fx*sc,(v-cy)/fy*sc,sc
  ]);
  const R=new THREE.Matrix3();
  R.set(pose[0],pose[1],pose[2],pose[4],pose[5],pose[6],pose[8],pose[9],pose[10]);
  const t=new THREE.Vector3(pose[3],pose[7],pose[11]);
  const o=t.clone();
  const wc=corners.map(c=>{const v=new THREE.Vector3(...c);v.applyMatrix3(R);v.add(t);return v;});
  const pts=[];
  for(const c of wc){pts.push(o.x,o.y,o.z,c.x,c.y,c.z);}
  for(let i=0;i<4;i++){const a=wc[i],b=wc[(i+1)%4];pts.push(a.x,a.y,a.z,b.x,b.y,b.z);}
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));
  frustumObj=new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:0xf85149,linewidth:2}));
  scene.add(frustumObj);
}

/* resize */
function resize(){
  const r=canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width,r.height);
  cam.aspect=r.width/r.height;
  cam.updateProjectionMatrix();
}
window.addEventListener('resize',resize);

/* render loop */
(function anim(){requestAnimationFrame(anim);ctrl.update();renderer.render(scene,cam);})();

/* ── image preloader ── */
function preImg(url){
  return new Promise((ok,no)=>{const i=new Image();i.onload=()=>ok(i.src);i.onerror=no;i.src=url;});
}

/* ── data loading ── */
async function loadScenes(){
  const list=await json('/api/scenes');
  scSel.innerHTML=list.map(s=>`<option value="${s}">${s.slice(0,8)}…</option>`).join('');
  if(list.length){S.sc=list[0];await loadEpisodes();}
}
async function loadEpisodes(){
  const list=await json(`/api/episodes/${S.sc}`);
  epSel.innerHTML=list.map(e=>`<option value="${e}">${e}</option>`).join('');
  if(list.length){S.ep=list[0];await loadMeta();}
}
async function loadMeta(){
  const m=await json(`/api/meta/${S.sc}/${S.ep}`);
  S.n=m.n_frames;S.K=m.K;
  frmR.max=S.n-1;frmR.value=0;S.f=0;
  updFrmLabel();
  await loadFrame(0);
}
function updFrmLabel(){frmV.textContent=`${S.f} / ${S.n-1}`;}

async function loadFrame(idx){
  const seq=++loadSeq;
  statusBadge.textContent='Loading…';statusBadge.style.color='var(--pr)';
  const q=`conf=${S.conf}&max_depth=${S.maxd}`;
  try{
    const [rSrc,dSrc,pBuf]=await Promise.all([
      preImg(`/api/rgb/${S.sc}/${S.ep}/${idx}`),
      preImg(`/api/depth/${S.sc}/${S.ep}/${idx}?${q}`),
      fetch(`/api/pcd/${S.sc}/${S.ep}/${idx}?downsample=${S.ds}&${q}`).then(r=>r.arrayBuffer()),
    ]);
    if(seq!==loadSeq)return;
    rgbImg.src=rSrc;depImg.src=dSrc;
    updatePcd(pBuf);
    statusBadge.textContent='Ready';statusBadge.style.color='var(--gn)';
  }catch(e){
    console.error(e);
    if(seq===loadSeq){statusBadge.textContent='Error';statusBadge.style.color='var(--rd)';}
  }
}

function updatePcd(buf){
  const dv=new DataView(buf);
  let off=0;
  const N=dv.getInt32(off,true);off+=4;
  const pose=new Float32Array(buf,off,16);off+=64;
  const K=new Float32Array(buf,off,4);off+=16;
  const pos=new Float32Array(buf,off,N*3);off+=N*12;
  const colU8=new Uint8Array(buf,off,N*3);
  const col=new Float32Array(N*3);
  for(let i=0;i<colU8.length;i++)col[i]=colU8[i]/255;

  pGeo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  pGeo.setAttribute('color',new THREE.BufferAttribute(col,3));
  pGeo.computeBoundingSphere();

  drawFrustum(pose,K);
  pcdInfo.textContent=`${N.toLocaleString()} pts`;
}

/* ── event handlers ── */
scSel.onchange=()=>{S.sc=scSel.value;S.playing=false;playBtn.textContent='▶';loadEpisodes();};
epSel.onchange=()=>{S.ep=epSel.value;S.playing=false;playBtn.textContent='▶';loadMeta();};

prevBtn.onclick=()=>{S.playing=false;playBtn.textContent='▶';S.f=Math.max(0,S.f-1);frmR.value=S.f;updFrmLabel();loadFrame(S.f);};
nextBtn.onclick=()=>{S.playing=false;playBtn.textContent='▶';S.f=Math.min(S.n-1,S.f+1);frmR.value=S.f;updFrmLabel();loadFrame(S.f);};
playBtn.onclick=()=>{S.playing=!S.playing;playBtn.textContent=S.playing?'⏸':'▶';if(S.playing)tick();};

frmR.oninput=()=>{S.f=+frmR.value;updFrmLabel();};
frmR.onchange=()=>{if(!S.playing)loadFrame(S.f);};

fpsR.oninput=()=>{S.fps=+fpsR.value;fpsV.textContent=S.fps.toFixed(1);};

confR.oninput=()=>{S.conf=+confR.value;confV.textContent=S.conf.toFixed(2);};
confR.onchange=()=>{if(!S.playing)loadFrame(S.f);};
maxdR.oninput=()=>{S.maxd=+maxdR.value;maxdV.textContent=S.maxd.toFixed(1);};
maxdR.onchange=()=>{if(!S.playing)loadFrame(S.f);};

dsR.oninput=()=>{S.ds=+dsR.value;dsV.textContent=S.ds;};
dsR.onchange=()=>{if(!S.playing)loadFrame(S.f);};
psR.oninput=()=>{const v=+psR.value/1000;S.ps=v;psV.textContent=v.toFixed(3);pMat.size=v;};

/* ── playback ── */
async function tick(){
  if(!S.playing)return;
  S.f=(S.f+1)%S.n;
  frmR.value=S.f;updFrmLabel();
  await loadFrame(S.f);
  if(S.playing)setTimeout(tick,1000/Math.max(.1,S.fps));
}

/* ── init ── */
resize();
loadScenes();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global BASE_DIR
    parser = argparse.ArgumentParser(description="Ropedia Web Viewer")
    parser.add_argument("--data-dir", type=str, default="datasets/ropedia/datasets",
                        help="Root directory containing scene UUID folders")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    BASE_DIR = Path(args.data_dir)
    assert BASE_DIR.exists(), f"Data directory not found: {BASE_DIR}"

    scenes = [d.name for d in BASE_DIR.iterdir() if d.is_dir()]
    print(f"Data root : {BASE_DIR.resolve()}")
    print(f"Scenes    : {len(scenes)}")
    print(f"Server    : http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
