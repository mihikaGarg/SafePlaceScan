"""SafePlaceScan web demo: upload a photo, get hazard detections."""
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

ROOT = Path(__file__).parent
model = YOLO(str(ROOT / "weights" / "best.pt"))


def scan(image, conf):
    if image is None:
        return None, []
    res = model.predict(image, conf=conf, verbose=False)[0]
    annotated = res.plot()[..., ::-1]  # BGR -> RGB
    found = [[res.names[int(b.cls)], round(float(b.conf), 2)] for b in res.boxes]
    return annotated, found


with gr.Blocks(title="SafePlaceScan") as demo:
    gr.Markdown("# SafePlaceScan\nFlags potholes, trash, graffiti, and tents/encampments in a photo.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Photo")
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
            btn = gr.Button("Scan", variant="primary")
        with gr.Column():
            out = gr.Image(label="Detections")
            table = gr.Dataframe(headers=["hazard", "confidence"], label="Found")
    examples = sorted(str(p) for p in (ROOT / "examples").glob("*.jpg"))
    if examples:
        gr.Examples(examples=[[e] for e in examples], inputs=inp)
    btn.click(scan, [inp, conf], [out, table])

if __name__ == "__main__":
    demo.launch()
