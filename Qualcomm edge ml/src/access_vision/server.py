from __future__ import annotations

import json
import logging
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np

from .board import build_http_notifier, build_notifier, build_ntfy_notifier
from .config import load_config
from .matching import AllowList
from .pipeline import FrameProcessor
from .runtime import QnnSession
from .vision import FaceDetector, FaceEmbedder, prepare_face

LOGGER = logging.getLogger(__name__)
MAX_FRAME_BYTES = 32 * 1024 * 1024


def _decode_rgba(body: bytes, width: int, height: int) -> np.ndarray:
    expected = width * height * 4
    if width <= 0 or height <= 0 or len(body) != expected:
        raise ValueError(f"Expected {expected} RGBA bytes for {width}x{height}, received {len(body)}")
    return np.frombuffer(body, dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()


class EnrollmentManager:
    def __init__(self, config, detector, embedder) -> None:
        self.config = config
        self.detector = detector
        self.embedder = embedder
        self.pending: dict[str, list[np.ndarray]] = {}

    def reset(self) -> dict:
        self.pending.clear()
        LOGGER.info("Enrollment batch reset")
        return {"status": "reset"}

    def add(self, person: str, frame_rgb: np.ndarray) -> dict:
        person = person.strip()
        if not person:
            raise ValueError("Person folder name is empty")
        faces = self.detector.detect(frame_rgb)
        if not faces:
            LOGGER.warning(
                "Enrollment skipped person=%s reason=no_face max_score=%.4f required=%.4f",
                person,
                self.detector.last_max_score,
                self.config.detector.score_threshold,
            )
            return {
                "status": "skipped",
                "reason": "no_face_above_threshold",
                "person": person,
                "maximum_detector_score": round(self.detector.last_max_score, 6),
                "required_score": self.config.detector.score_threshold,
                "detector": self.config.detector.model_id,
            }
        face = max(
            faces,
            key=lambda item: (item.xyxy[2] - item.xyxy[0]) * (item.xyxy[3] - item.xyxy[1]),
        )
        embedding = self.embedder.embed(
            prepare_face(frame_rgb, face, self.config.embedder)
        )
        self.pending.setdefault(person, []).append(embedding)
        LOGGER.info(
            "Enrollment accepted person=%s templates_in_batch=%d bbox=%s",
            person,
            len(self.pending[person]),
            face.xyxy,
        )
        return {
            "status": "accepted",
            "person": person,
            "count": len(self.pending[person]),
            "bbox": list(face.xyxy),
            "landmarks": (
                [list(point) for point in face.landmarks]
                if face.landmarks is not None
                else []
            ),
        }

    def commit(self) -> dict:
        minimum = self.config.enrollment.minimum_images_per_person
        database: dict[str, list[list[float]]] = {}
        skipped: dict[str, int] = {}
        for person, embeddings in self.pending.items():
            if len(embeddings) < minimum:
                skipped[person] = len(embeddings)
                continue
            templates: list[list[float]] = []
            for embedding in embeddings:
                vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(vector))
                if norm < 1e-12:
                    raise ValueError(f"Zero embedding generated for identity {person!r}")
                templates.append((vector / norm).astype(float).tolist())
            database[person] = templates
        if not database:
            raise RuntimeError(f"No identity has at least {minimum} accepted images")

        output = self.config.database.embeddings_path
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(database, handle, indent=2)
            handle.write("\n")
        temporary.replace(output)
        LOGGER.info(
            "Embedding database saved path=%s identities=%d templates=%d skipped=%s",
            output,
            len(database),
            sum(len(items) for items in database.values()),
            skipped,
        )
        return {
            "status": "saved",
            "identities": sorted(database),
            "templates": sum(len(items) for items in database.values()),
            "skipped": skipped,
            "path": str(output),
        }


def _legacy_live_html(config) -> str:
    settings = json.dumps({
        "width": config.web.frame_width,
        "height": config.web.frame_height,
        "interval": config.web.frame_interval_ms,
        "cameras": [
            {"id": camera.id, "source": camera.source}
            for camera in config.cameras
        ],
    })
    return f"""<!doctype html><html><head><meta charset=utf-8><title>Access Vision</title>
<style>body{{font:16px system-ui;max-width:850px;margin:30px auto;padding:0 16px}}canvas{{display:block;width:100%;max-height:560px;margin-top:12px;background:#111}}button,select{{font:inherit;padding:8px}}#status{{padding:12px;margin-top:10px;background:#eee;white-space:pre-wrap}}</style></head>
<body><h1>Snapdragon Access Vision</h1><p>Frames stay on this laptop and are sent only to the localhost NPU service.</p>
<label>Camera <select id=camera></select></label> <button id=start>Start camera</button>
<video id=video autoplay muted playsinline hidden></video><img id=remote alt="Remote camera" hidden><canvas id=canvas></canvas><div id=status>Stopped</div>
<script>const cfg={settings}; const video=document.querySelector('#video'), remote=document.querySelector('#remote'), canvas=document.querySelector('#canvas');
const status=document.querySelector('#status'), camera=document.querySelector('#camera'); cfg.cameras.forEach(x=>camera.add(new Option(`${{x.id}} (${{x.source}})`,x.id)));
let frameSource=null;
let running=false; document.querySelector('#start').onclick=async()=>{{
 if(running)return; try{{const selected=cfg.cameras.find(x=>x.id===camera.value);
  if(selected.source==='webcam'){{video.srcObject=await navigator.mediaDevices.getUserMedia({{video:{{width:{{ideal:cfg.width}},height:{{ideal:cfg.height}},frameRate:{{ideal:40,max:40}}}},audio:false}});await video.play();frameSource=video;}}
  else{{frameSource=remote;remote.onload=()=>{{if(running)status.textContent='Remote camera connected; waiting for inference.';}};
   remote.onerror=()=>{{running=false;frameSource=null;camera.disabled=false;status.textContent=`Camera error: ${{selected.id}} stream is unavailable or already open in another DroidCam client. Close any direct /video tab or other client, then click Start camera again.`;}};
   remote.src='/api/camera-stream?'+new URLSearchParams({{camera:selected.id,attempt:Date.now()}});}}
  camera.disabled=true;running=true;status.textContent=`Connecting to ${{selected.id}}...`;tick();}}
 catch(e){{status.textContent='Camera error: '+e;}}
}};
function drawResult(c,out){{
 for(const face of (out.faces||[])){{
  const [x1,y1,x2,y2]=face.bbox, color=face.allowed?'#16c784':'#ff3b30';
  c.strokeStyle=color;c.lineWidth=3;c.strokeRect(x1,y1,x2-x1,y2-y1);
  const identity=face.allowed?(face.person_id||'allowed'):'UNAUTHORIZED';
  const label=`${{identity}} ${{Number(face.cosine_similarity).toFixed(3)}}`;
  c.font='bold 18px system-ui';const tw=c.measureText(label).width,pad=5;
  const labelY=Math.max(0,y1-28);c.fillStyle=color;c.fillRect(x1,labelY,tw+pad*2,28);
  c.fillStyle='#fff';c.fillText(label,x1+pad,labelY+20);
  c.fillStyle='#00d9ff';for(const point of (face.landmarks||[])){{c.beginPath();c.arc(point[0],point[1],4,0,Math.PI*2);c.fill();}}
 }}
}}
async function tick(){{if(!running)return;const started=performance.now();
 if(frameSource===remote&&(!remote.naturalWidth||!remote.naturalHeight)){{status.textContent='Connecting to remote camera; waiting for the first MJPEG frame...';setTimeout(tick,250);return;}}
 if(frameSource===video&&video.readyState<2){{status.textContent='Waiting for the first webcam frame...';setTimeout(tick,100);return;}}
 try{{canvas.width=cfg.width;canvas.height=cfg.height;const c=canvas.getContext('2d',{{willReadFrequently:true}});c.drawImage(frameSource,0,0,cfg.width,cfg.height);const pixels=c.getImageData(0,0,cfg.width,cfg.height).data;
 const q=new URLSearchParams({{camera:camera.value,width:cfg.width,height:cfg.height}});const r=await fetch('/api/frame?'+q,{{method:'POST',headers:{{'Content-Type':'application/octet-stream'}},body:pixels}});const out=await r.json();drawResult(c,out);status.textContent=JSON.stringify(out,null,2);
 }}catch(e){{status.textContent='Processing error: '+e;}}
 const remaining=Math.max(0,cfg.interval-(performance.now()-started));setTimeout(tick,remaining);}}
</script></body></html>"""


def _live_html(config) -> str:
    settings = json.dumps({
        "width": config.web.frame_width,
        "height": config.web.frame_height,
        "interval": config.web.frame_interval_ms,
        "target_fps": round(1000 / config.web.frame_interval_ms),
        "detector": config.detector.model_id,
        "embedder": config.embedder.model_id,
        "threshold": config.database.cosine_threshold,
        "cameras": [
            {"id": camera.id, "source": camera.source}
            for camera in config.cameras
        ],
    })
    template = Path(__file__).with_name("live_ui.html").read_text(encoding="utf-8")
    return template.replace("__ACCESS_VISION_SETTINGS__", settings)


def _enroll_html(config) -> str:
    return f"""<!doctype html><html><head><meta charset=utf-8><title>Enroll Faces</title>
<style>body{{font:16px system-ui;max-width:850px;margin:30px auto;padding:0 16px}}button,input{{font:inherit;padding:8px}}#status{{padding:12px;margin-top:10px;background:#eee;white-space:pre-wrap}}</style></head>
<body><h1>Enroll allowed people</h1><p>Select the <code>allowed_people</code> folder containing one subfolder per person and at least {config.enrollment.minimum_images_per_person} images per person.</p>
<input id=folder type=file webkitdirectory multiple accept="image/*"> <button id=run>Generate embeddings</button>
<canvas id=canvas hidden></canvas><div id=status>Waiting for a folder.</div>
<script>const status=document.querySelector('#status'), canvas=document.querySelector('#canvas');
document.querySelector('#run').onclick=async()=>{{const files=[...document.querySelector('#folder').files].filter(f=>f.type.startsWith('image/'));if(!files.length){{status.textContent='Choose a folder containing images first.';return;}}
 await fetch('/api/enroll/reset',{{method:'POST'}});let accepted=0,skipped=0,failures=[];
 for(let i=0;i<files.length;i++){{const f=files[i],parts=f.webkitRelativePath.split('/'),person=parts.length>1?parts[parts.length-2]:'';status.textContent=`Processing ${{i+1}}/${{files.length}}: ${{f.webkitRelativePath}}`;
  try{{const bitmap=await createImageBitmap(f),scale=Math.min(1,1280/Math.max(bitmap.width,bitmap.height)),w=Math.max(1,Math.round(bitmap.width*scale)),h=Math.max(1,Math.round(bitmap.height*scale));canvas.width=w;canvas.height=h;const c=canvas.getContext('2d',{{willReadFrequently:true}});c.drawImage(bitmap,0,0,w,h);bitmap.close();const pixels=c.getImageData(0,0,w,h).data;
   const q=new URLSearchParams({{person,width:w,height:h}});const r=await fetch('/api/enroll/frame?'+q,{{method:'POST',headers:{{'Content-Type':'application/octet-stream'}},body:pixels}});const out=await r.json();if(out.status==='accepted')accepted++;else{{skipped++;failures.push({{file:f.webkitRelativePath,...out}});}}
  }}catch(e){{skipped++;failures.push({{file:f.webkitRelativePath,error:String(e)}});}}
 }} const r=await fetch('/api/enroll/commit',{{method:'POST'}}),out=await r.json();status.textContent=JSON.stringify({{accepted,skipped,failures,result:out}},null,2);
}};</script></body></html>"""


def _build_notifier(config, processor) -> object | None:
    if processor is None or not config.board.enabled:
        return None
    if config.board.transport == "ntfy":
        if not config.board.ntfy_topic:
            raise RuntimeError("board.transport is ntfy but board.ntfy_topic is empty")
        return build_ntfy_notifier(
            config.board.ntfy_topic,
            base_url=config.board.ntfy_base_url,
            heartbeat_seconds=config.board.heartbeat_seconds,
            min_interval_seconds=config.board.min_interval_seconds,
        )
    if config.board.transport == "http":
        return build_http_notifier(
            config.board.url,
            config.board.token,
            heartbeat_seconds=config.board.heartbeat_seconds,
            min_interval_seconds=config.board.min_interval_seconds,
        )
    return build_notifier(
        scripts_dir=str(config.board.scripts_dir) if config.board.scripts_dir else None,
        heartbeat_seconds=config.board.heartbeat_seconds,
        min_interval_seconds=config.board.min_interval_seconds,
    )


def _build_live_runtime(config) -> dict:
    LOGGER.info(
        "Loading live runtime detector=%s embedder=%s embedding_dimension=%d alignment=%s threshold=%.3f",
        config.detector.model_id,
        config.embedder.model_id,
        config.embedder.embedding_dimension,
        config.embedder.align_landmarks,
        config.database.cosine_threshold,
    )
    LOGGER.info("Detector path: %s", config.detector.path)
    if config.detector.landmark_path is not None:
        LOGGER.info("Detector landmark path: %s", config.detector.landmark_path)
    LOGGER.info("Embedder path: %s", config.embedder.path)
    landmark_session = (
        QnnSession(config.detector.landmark_path, config.runtime)
        if config.detector.landmark_path is not None
        else None
    )
    detector = FaceDetector(
        QnnSession(config.detector.path, config.runtime),
        config.detector,
        landmark_session,
    )
    embedder = FaceEmbedder(
        QnnSession(config.embedder.path, config.runtime), config.embedder
    )
    allow_list = AllowList.load(
        config.database.embeddings_path,
        config.database.cosine_threshold,
        config.embedder.embedding_dimension,
    )
    LOGGER.info(
        "Loaded embeddings path=%s identities=%d templates=%d dimension=%d",
        config.database.embeddings_path,
        len(set(allow_list.names)),
        len(allow_list.names),
        allow_list.embeddings.shape[1],
    )
    processor = FrameProcessor(config, detector, embedder, allow_list)
    notifier = _build_notifier(config, processor)
    LOGGER.info("Board output %s", "enabled" if notifier else "unavailable/disabled")
    return {
        "config": config,
        "processor": processor,
        "notifier": notifier,
        "page": _live_html(config),
        "cameras_by_id": {camera.id: camera for camera in config.cameras},
    }


def run_server(
    config,
    enrollment_only: bool,
    config_path: str | Path | None = None,
) -> None:
    if config.web.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The raw-camera server must bind to localhost only")
    LOGGER.info(
        "Pipeline mode=%s detector=%s embedder=%s embedding_dimension=%d alignment=%s threshold=%.3f",
        "enrollment" if enrollment_only else "live",
        config.detector.model_id,
        config.embedder.model_id,
        config.embedder.embedding_dimension,
        config.embedder.align_landmarks,
        config.database.cosine_threshold,
    )
    lock = threading.Lock()
    stop_reload = threading.Event()
    enrollment = None
    if enrollment_only:
        LOGGER.info("Detector path: %s", config.detector.path)
        if config.detector.landmark_path is not None:
            LOGGER.info("Detector landmark path: %s", config.detector.landmark_path)
        LOGGER.info("Embedder path: %s", config.embedder.path)
        landmark_session = (
            QnnSession(config.detector.landmark_path, config.runtime)
            if config.detector.landmark_path is not None
            else None
        )
        detector = FaceDetector(
            QnnSession(config.detector.path, config.runtime),
            config.detector,
            landmark_session,
        )
        embedder = FaceEmbedder(
            QnnSession(config.embedder.path, config.runtime), config.embedder
        )
        enrollment = EnrollmentManager(config, detector, embedder)
        state = {
            "config": config,
            "processor": None,
            "notifier": None,
            "page": _enroll_html(config),
            "cameras_by_id": {camera.id: camera for camera in config.cameras},
        }
        LOGGER.info(
            "Enrollment output path=%s; existing database is replaced only after a successful commit",
            config.database.embeddings_path,
        )
    else:
        state = _build_live_runtime(config)

    def reload_loop(path: Path) -> None:
        try:
            last_stamp = path.stat().st_mtime_ns
        except OSError:
            last_stamp = 0
        while not stop_reload.wait(1.0):
            try:
                stamp = path.stat().st_mtime_ns
            except OSError as exc:
                LOGGER.warning("Config hot reload skipped; cannot stat %s: %s", path, exc)
                continue
            if stamp == last_stamp:
                continue
            try:
                new_config = load_config(path)
                if new_config.web != state["config"].web:
                    LOGGER.warning(
                        "Config hot reload ignores [web] changes; restart required for host/port/frame size"
                    )
                new_state = _build_live_runtime(new_config)
            except Exception as exc:  # noqa: BLE001 - keep the known-good runtime alive
                LOGGER.exception("Config hot reload failed; keeping previous runtime: %s", exc)
                last_stamp = stamp
                continue
            old_notifier = state.get("notifier")
            with lock:
                state.update(new_state)
            if old_notifier is not None:
                old_notifier.close()
            last_stamp = stamp
            LOGGER.info(
                "Hot reloaded config detector=%s path=%s",
                new_config.detector.model_id,
                new_config.detector.path,
            )

    if config_path is not None and not enrollment_only:
        reload_path = Path(config_path).resolve()
        threading.Thread(
            target=reload_loop,
            args=(reload_path,),
            name="config-hot-reload",
            daemon=True,
        ).start()
        LOGGER.info("Config hot reload enabled path=%s interval=1.0s", reload_path)

    # Camera endpoints are LAN addresses. Do not send them through machine-wide
    # HTTP proxies, which commonly cannot route private IP addresses.
    camera_opener = build_opener(ProxyHandler({}))

    class Handler(BaseHTTPRequestHandler):
        def _json(self, value: dict, status: int = 200) -> None:
            payload = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._json({"status": "ok", "mode": "enrollment" if enrollment_only else "live"})
                return
            if parsed.path == "/api/camera-stream" and not enrollment_only:
                query = parse_qs(parsed.query)
                camera_id = query.get("camera", [""])[0]
                with lock:
                    camera = state["cameras_by_id"].get(camera_id)
                if camera is None or camera.source != "mjpeg" or not camera.url:
                    self._json({"error": "Unknown MJPEG camera"}, 404)
                    return
                response_started = False
                try:
                    request = Request(camera.url, headers={"User-Agent": "AccessVision/0.2"})
                    with camera_opener.open(request, timeout=30) as upstream:
                        content_type = upstream.headers.get(
                            "Content-Type", "multipart/x-mixed-replace"
                        )
                        if not (
                            content_type.lower().startswith("multipart/")
                            or content_type.lower().startswith("image/")
                        ):
                            message = upstream.read(4096).decode("utf-8", "replace")
                            if "DroidCam busy" in message:
                                raise RuntimeError(
                                    "DroidCam is connected to another client; "
                                    "close the direct /video browser tab or other DroidCam client"
                                )
                            raise RuntimeError(
                                f"Expected an MJPEG stream, received {content_type}"
                            )
                        self.send_response(200)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        response_started = True
                        try:
                            while chunk := upstream.read(64 * 1024):
                                self.wfile.write(chunk)
                                self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, TimeoutError):
                            pass
                except Exception as exc:
                    LOGGER.warning("MJPEG camera %s failed: %s", camera_id, exc)
                    if not response_started and not self.wfile.closed:
                        try:
                            self._json({"error": f"Camera stream unavailable: {exc}"}, 502)
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                return
            with lock:
                payload = state["page"].encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            try:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_FRAME_BYTES:
                    raise ValueError("Frame is too large")
                body = self.rfile.read(length)
                with lock:
                    processor = state["processor"]
                    notifier = state["notifier"]
                    if parsed.path == "/api/frame" and processor is not None:
                        frame = _decode_rgba(body, int(query["width"][0]), int(query["height"][0]))
                        result = processor.process(query["camera"][0], frame)
                        if notifier is not None:
                            notifier.update(result)
                        self._json(result)
                    elif parsed.path == "/api/enroll/reset" and enrollment is not None:
                        self._json(enrollment.reset())
                    elif parsed.path == "/api/enroll/frame" and enrollment is not None:
                        frame = _decode_rgba(body, int(query["width"][0]), int(query["height"][0]))
                        self._json(enrollment.add(query["person"][0], frame))
                    elif parsed.path == "/api/enroll/commit" and enrollment is not None:
                        self._json(enrollment.commit())
                    else:
                        self._json({"error": "not_found"}, 404)
            except Exception as exc:
                LOGGER.exception("Request failed")
                self._json({"error": str(exc)}, 400)

        def log_message(self, format: str, *args) -> None:
            LOGGER.debug(format, *args)

    server = ThreadingHTTPServer((config.web.host, config.web.port), Handler)
    server.daemon_threads = True
    url = f"http://{config.web.host}:{config.web.port}/"
    LOGGER.info("Local %s UI: %s", "enrollment" if enrollment_only else "camera", url)
    if config.web.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping")
    finally:
        stop_reload.set()
        notifier = state.get("notifier")
        if notifier is not None:
            notifier.close()
        server.server_close()
