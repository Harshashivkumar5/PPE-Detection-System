"""
PPE Detection System - Gradio App (FIXED & PROFESSIONAL)
==========================================================
Fixes applied:
1. No more hardcoded Windows paths
2. Fixed model loading with auto-discovery
3. Proper webcam streaming with state management
4. Alert logic now correctly identifies missing PPE
5. Never shows "No items detected" when person is present
6. Professional dark theme UI with status cards
7. Confidence display per detection
8. Video file support
9. All modes work: image upload, webcam, video
"""

import cv2
import numpy as np
import gradio as gr
from pathlib import Path
import time
import json
from ppe_detector import PPEDetector

# ─── INITIALIZE ───────────────────────────────────────────────────────────────
print("Initializing PPE Detector...")
detector = PPEDetector(conf=0.35, iou=0.45)

PROJECT_ROOT = Path(__file__).parent

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def format_detections_html(detections: list) -> str:
    """Format detection list as styled HTML cards."""
    if not detections:
        return "<div class='no-det'>No detections</div>"
    
    html = "<div class='det-list'>"
    for d in detections:
        cls   = d["class_name"]
        conf  = d["confidence"]
        
        # Color coding
        if cls in ("jacket", "hat"):
            card_cls = "det-present"
            icon = "✅"
        elif cls in ("no-jacket", "no-hat"):
            card_cls = "det-absent"
            icon = "❌"
        else:
            card_cls = "det-neutral"
            icon = "👤"
        
        conf_pct = int(conf * 100)
        html += f"""
        <div class='det-card {card_cls}'>
          <span class='det-icon'>{icon}</span>
          <span class='det-name'>{cls.upper()}</span>
          <div class='conf-bar-wrap'>
            <div class='conf-bar' style='width:{conf_pct}%'></div>
          </div>
          <span class='conf-val'>{conf_pct}%</span>
        </div>"""
    html += "</div>"
    return html


def format_alerts_html(alerts: list, all_clear: bool) -> str:
    """Format alerts as styled HTML."""
    if all_clear and not alerts:
        return "<div class='alert-ok'>✅ All PPE Requirements Met</div>"
    
    html = "<div class='alerts'>"
    for a in alerts:
        cls = "alert-critical" if "NOT" in a or "🚨" in a else "alert-warning"
        html += f"<div class='{cls}'>{a}</div>"
    html += "</div>"
    return html


def make_ppe_status_html(detections: list) -> str:
    """Create PPE status cards for hat, jacket, shoes."""
    detected_names = {d["class_name"] for d in detections}
    
    items = [
        ("🪖", "Hat",    "hat",    "no-hat"),
        ("🦺", "Jacket", "jacket", "no-jacket"),
        ("👟", "Shoes",  None,     None),   # Not in dataset
    ]
    
    html = "<div class='ppe-status-grid'>"
    for icon, label, present_cls, absent_cls in items:
        if present_cls in detected_names:
            status = "present"
            status_text = "WEARING"
            bg = "#1a3a1a"
            border = "#2ecc40"
        elif absent_cls in detected_names:
            status = "absent"
            status_text = "NOT WEARING"
            bg = "#3a1a1a"
            border = "#ff4136"
        else:
            status = "unknown"
            status_text = "NOT DETECTED"
            bg = "#1a1a2e"
            border = "#555"
        
        html += f"""
        <div class='ppe-card' style='background:{bg};border-color:{border}'>
          <div class='ppe-icon'>{icon}</div>
          <div class='ppe-label'>{label}</div>
          <div class='ppe-status ppe-{status}'>{status_text}</div>
        </div>"""
    html += "</div>"
    return html


# ─── DETECTION FUNCTIONS ──────────────────────────────────────────────────────

def detect_image(image: np.ndarray):
    """Detect PPE in uploaded image."""
    if image is None:
        return None, "<div class='no-det'>Please upload an image</div>", \
               "<div class='alert-ok'>—</div>", make_ppe_status_html([])
    
    result = detector.detect(image)
    
    det_html    = format_detections_html(result["detections"])
    alerts_html = format_alerts_html(result["alerts"], result["all_clear"])
    status_html = make_ppe_status_html(result["detections"])
    
    return result["annotated_frame"], det_html, alerts_html, status_html


def detect_webcam(frame: np.ndarray):
    """Real-time webcam detection (streaming mode)."""
    if frame is None:
        return None, "<div class='no-det'>No frame</div>", \
               "<div class='alert-ok'>Waiting...</div>", make_ppe_status_html([])
    
    result = detector.detect(frame)
    
    det_html    = format_detections_html(result["detections"])
    alerts_html = format_alerts_html(result["alerts"], result["all_clear"])
    status_html = make_ppe_status_html(result["detections"])
    
    return result["annotated_frame"], det_html, alerts_html, status_html


def detect_video(video_path: str):
    """Process uploaded video file."""
    if video_path is None:
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_path = str(PROJECT_ROOT / "runs" / "video_output.mp4")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.detect(frame)
        out.write(result["annotated_frame"])
        frame_count += 1
        
        if frame_count > 3000:  # Safety limit
            break
    
    cap.release()
    out.release()
    return out_path


# ─── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
/* ── Global ──────────────────────────────────── */
:root {
  --bg:       #0d0d1a;
  --surface:  #12122a;
  --surface2: #1a1a35;
  --accent:   #3a7bd5;
  --green:    #2ecc40;
  --red:      #ff4136;
  --orange:   #ff851b;
  --text:     #e0e0f0;
  --subtext:  #888aaa;
}
body, .gradio-container { background: var(--bg) !important; color: var(--text) !important; }
.gradio-container { max-width: 1400px !important; margin: 0 auto; }

/* ── Header ──────────────────────────────────── */
.app-header {
  background: linear-gradient(135deg, #0d1b3e 0%, #1a0a2e 100%);
  border: 1px solid #2a2a5a;
  border-radius: 12px;
  padding: 24px 32px;
  margin-bottom: 20px;
  text-align: center;
}
.app-title { font-size: 2rem; font-weight: 700; color: #fff;
  background: linear-gradient(90deg, #3a7bd5, #7b5ea7);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.app-subtitle { color: var(--subtext); font-size: 0.95rem; margin-top: 4px; }

/* ── Tabs ─────────────────────────────────────── */
.tab-nav button { background: var(--surface2) !important; color: var(--subtext) !important;
  border: 1px solid #2a2a5a !important; border-radius: 8px !important; padding: 8px 20px !important; }
.tab-nav button.selected { background: var(--accent) !important; color: #fff !important; }

/* ── Cards ─────────────────────────────────────── */
.panel-card {
  background: var(--surface); border: 1px solid #2a2a5a;
  border-radius: 10px; padding: 16px;
}
.panel-title {
  font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 1px; color: var(--subtext); margin-bottom: 10px;
}

/* ── PPE Status Grid ─────────────────────────── */
.ppe-status-grid {
  display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px;
}
.ppe-card {
  flex: 1; min-width: 90px; border: 1.5px solid;
  border-radius: 10px; padding: 12px 8px; text-align: center;
  transition: transform 0.2s;
}
.ppe-card:hover { transform: translateY(-2px); }
.ppe-icon { font-size: 1.6rem; }
.ppe-label { font-size: 0.75rem; font-weight: 600; color: var(--subtext);
  text-transform: uppercase; letter-spacing: 0.5px; margin: 4px 0; }
.ppe-status { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.5px; }
.ppe-present { color: #2ecc40; }
.ppe-absent  { color: #ff4136; }
.ppe-unknown { color: #888; }

/* ── Detection Cards ─────────────────────────── */
.det-list { display: flex; flex-direction: column; gap: 6px; }
.det-card {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 10px; border-radius: 8px; border: 1px solid;
}
.det-present { background: #0d2b0d; border-color: #2ecc40; }
.det-absent  { background: #2b0d0d; border-color: #ff4136; }
.det-neutral { background: #1a1a2e; border-color: #444; }
.det-icon  { font-size: 1rem; }
.det-name  { font-weight: 600; font-size: 0.75rem; flex: 1; }
.conf-bar-wrap { width: 60px; height: 6px; background: #333; border-radius: 3px; }
.conf-bar { height: 6px; background: var(--accent); border-radius: 3px; }
.conf-val { font-size: 0.7rem; color: var(--subtext); min-width: 30px; text-align: right; }
.no-det { color: var(--subtext); font-size: 0.85rem; padding: 10px; text-align: center; }

/* ── Alerts ──────────────────────────────────── */
.alerts { display: flex; flex-direction: column; gap: 6px; }
.alert-ok {
  background: #0d2b0d; border: 1.5px solid #2ecc40;
  color: #2ecc40; border-radius: 8px; padding: 10px 14px;
  font-weight: 600; font-size: 0.85rem;
}
.alert-critical {
  background: #2b0d0d; border: 1.5px solid #ff4136;
  color: #ff4136; border-radius: 8px; padding: 10px 14px;
  font-weight: 600; font-size: 0.85rem; animation: pulse 1.5s infinite;
}
.alert-warning {
  background: #2b1a0d; border: 1.5px solid #ff851b;
  color: #ff851b; border-radius: 8px; padding: 10px 14px;
  font-weight: 600; font-size: 0.85rem;
}
@keyframes pulse {
  0%, 100% { opacity: 1; } 50% { opacity: 0.7; }
}

/* ── Buttons ─────────────────────────────────── */
.gr-button { 
  background: linear-gradient(135deg, var(--accent), #5a4bd5) !important;
  border: none !important; color: white !important; font-weight: 600 !important;
  border-radius: 8px !important; padding: 10px 24px !important;
}
.gr-button:hover { opacity: 0.9; transform: translateY(-1px); }

/* ── Fix Gradio dark overrides ─────────────────── */
.dark { --background-fill-primary: var(--bg) !important; }
label { color: var(--subtext) !important; }
"""

# ─── GRADIO UI ────────────────────────────────────────────────────────────────

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
    ),
    css=CSS,
    title="PPE Detection System",
) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class='app-header'>
      <div class='app-title'>🦺 PPE Detection System</div>
      <div class='app-subtitle'>
        Real-Time Personal Protective Equipment Detection &nbsp;|&nbsp;
        Jacket · Hat · Safety Compliance
      </div>
    </div>
    """)

    # ── Tabs ────────────────────────────────────────────────────────────────
    with gr.Tabs() as tabs:

        # ── IMAGE UPLOAD TAB ────────────────────────────────────────────────
        with gr.TabItem("📷 Image Upload"):
            with gr.Row():
                with gr.Column(scale=3):
                    img_input = gr.Image(
                        type="numpy",
                        label="Upload Image",
                        height=480,
                    )
                    detect_btn = gr.Button("🔍 Detect PPE", variant="primary")
                
                with gr.Column(scale=3):
                    img_output = gr.Image(label="Detection Output", height=480)
                
                with gr.Column(scale=2):
                    gr.HTML("<div class='panel-title'>PPE Status</div>")
                    img_ppe_status = gr.HTML(make_ppe_status_html([]))
                    
                    gr.HTML("<div class='panel-title' style='margin-top:12px'>Alerts</div>")
                    img_alerts = gr.HTML("<div class='alert-ok'>—</div>")
                    
                    gr.HTML("<div class='panel-title' style='margin-top:12px'>Detections</div>")
                    img_detections = gr.HTML("<div class='no-det'>—</div>")

            detect_btn.click(
                fn=detect_image,
                inputs=img_input,
                outputs=[img_output, img_detections, img_alerts, img_ppe_status],
            )
            img_input.change(
                fn=detect_image,
                inputs=img_input,
                outputs=[img_output, img_detections, img_alerts, img_ppe_status],
            )

        # ── WEBCAM TAB ──────────────────────────────────────────────────────
        with gr.TabItem("🎥 Live Webcam"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML("<div class='panel-title'>Live Camera Feed</div>")
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="Webcam",
                        height=480,
                    )
                
                with gr.Column(scale=3):
                    gr.HTML("<div class='panel-title'>Live Detection</div>")
                    webcam_output = gr.Image(label="Live Output", height=480)
                
                with gr.Column(scale=2):
                    gr.HTML("<div class='panel-title'>PPE Status (Live)</div>")
                    cam_ppe_status = gr.HTML(make_ppe_status_html([]))
                    
                    gr.HTML("<div class='panel-title' style='margin-top:12px'>Live Alerts</div>")
                    cam_alerts = gr.HTML("<div class='alert-ok'>Waiting for stream...</div>")
                    
                    gr.HTML("<div class='panel-title' style='margin-top:12px'>Detections</div>")
                    cam_detections = gr.HTML("<div class='no-det'>—</div>")

            gr.HTML("""
            <div style='color:#888;font-size:0.8rem;padding:8px 0;'>
              ℹ️ Click "Start Webcam" button in the video component above to begin live detection.
              Ensure your browser has camera permissions.
            </div>
            """)

            webcam_input.stream(
                fn=detect_webcam,
                inputs=webcam_input,
                outputs=[webcam_output, cam_detections, cam_alerts, cam_ppe_status],
                time_limit=300,
                stream_every=0.1,  # ~10 FPS
            )

        # ── VIDEO UPLOAD TAB ─────────────────────────────────────────────────
        with gr.TabItem("🎬 Video Upload"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video")
                    vid_btn = gr.Button("⚙️ Process Video", variant="primary")
                    gr.HTML("""
                    <div style='color:#888;font-size:0.8rem;margin-top:8px;'>
                      Supports MP4, AVI, MOV, MKV. Processing time depends on video length.
                    </div>
                    """)
                
                with gr.Column(scale=1):
                    video_output = gr.Video(label="Processed Video")

            vid_btn.click(
                fn=detect_video,
                inputs=video_input,
                outputs=video_output,
            )

        # ── INFO TAB ────────────────────────────────────────────────────────
        with gr.TabItem("ℹ️ Info"):
            gr.HTML("""
            <div style='padding: 20px; color: #ccc; line-height: 1.8;'>
              <h3 style='color:#fff;'>PPE Detection System — Class Information</h3>
              
              <table style='border-collapse:collapse; width:100%; margin-top:16px;'>
                <tr style='background:#1a1a35; color:#aaa; font-size:0.8rem;'>
                  <th style='padding:8px 12px; border:1px solid #333; text-align:left;'>Class ID</th>
                  <th style='padding:8px 12px; border:1px solid #333; text-align:left;'>Name</th>
                  <th style='padding:8px 12px; border:1px solid #333; text-align:left;'>Description</th>
                  <th style='padding:8px 12px; border:1px solid #333; text-align:left;'>Color</th>
                </tr>
                <tr><td style='padding:8px 12px;border:1px solid #333'>0</td>
                    <td style='color:#2ecc40'>jacket</td>
                    <td>Safety vest / hi-vis jacket present</td>
                    <td>🟢 Green</td></tr>
                <tr><td style='padding:8px 12px;border:1px solid #333'>1</td>
                    <td style='color:#3af'>hat</td>
                    <td>Hardhat / safety helmet present</td>
                    <td>🔵 Blue</td></tr>
                <tr><td style='padding:8px 12px;border:1px solid #333'>2</td>
                    <td style='color:#f44'>no-jacket</td>
                    <td>Person explicitly NOT wearing jacket</td>
                    <td>🔴 Red</td></tr>
                <tr><td style='padding:8px 12px;border:1px solid #333'>3</td>
                    <td style='color:#f44'>no-hat</td>
                    <td>Person explicitly NOT wearing hardhat</td>
                    <td>🔴 Dark Red</td></tr>
                <tr><td style='padding:8px 12px;border:1px solid #333'>4</td>
                    <td style='color:#aaa'>person</td>
                    <td>Person detected (generic)</td>
                    <td>⚫ Grey</td></tr>
              </table>
              
              <h3 style='color:#fff; margin-top:24px;'>Alert Logic</h3>
              <ul>
                <li><b style='color:#f44'>⛔ NOT wearing</b> — Model explicitly detected an absence class</li>
                <li><b style='color:#ff851b'>⚠️ Not visible</b> — Person detected but PPE not seen (may be out of frame)</li>
                <li><b style='color:#2ecc40'>✅ All clear</b> — All visible PPE items detected as present</li>
              </ul>

              <h3 style='color:#fff; margin-top:24px;'>Note on Shoes</h3>
              <p>The training dataset does not include a "shoes" class. The original 
              Roboflow dataset focused on hardhats and safety vests. To add shoe detection,
              annotate shoe data and retrain with class ID 5 added.</p>
            </div>
            """)

    # ── Footer ───────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style='text-align:center; color:#444; font-size:0.75rem; padding:16px 0; margin-top:8px;
                border-top:1px solid #1a1a35;'>
      PPE Detection System &nbsp;·&nbsp; YOLOv8 + Ultralytics
      &nbsp;·&nbsp; Confidence threshold: 35%
    </div>
    """)


# ─── LAUNCH ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
    )
