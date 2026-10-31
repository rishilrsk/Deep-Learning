"""Minimal image-captioning web app."""

from functools import lru_cache
import os

import gradio as gr
import torch
from PIL import Image, ImageOps
from transformers import BlipForConditionalGeneration, BlipProcessor


MODEL_ID = "Salesforce/blip-image-captioning-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMAGE_SIDE = 1600
torch.set_num_threads(min(8, os.cpu_count() or 4))


@lru_cache(maxsize=1)
def load_captioner():
    """Load the captioning model once and reuse it for later uploads."""
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    model.to(DEVICE).eval()
    return processor, model


def generate_caption(image: Image.Image) -> str:
    if image is None:
        raise gr.Error("Upload an image first.")

    try:
        processor, model = load_captioner()
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=40, num_beams=3)
        caption = processor.decode(output[0], skip_special_tokens=True).strip()
        return caption or "I couldn't generate a caption for this image."
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(
            "Captioning couldn't start. Check that the model can be downloaded "
            "and see the app logs for details."
        ) from exc


CSS = """
/* ── Google Font import ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Reset & base ── */
html, body {
  background: #f8f9fc !important;
  color-scheme: light !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
html, body, gradio-app, #root {
  width: 100% !important;
  min-height: 100% !important;
  margin: 0 !important;
}

/* ── Container tokens ── */
.gradio-container {
  width: 100% !important;
  max-width: none !important;
  min-height: 100vh !important;
  margin: 0 !important;
  padding: 0 clamp(16px, 5vw, 80px) !important;
  box-sizing: border-box !important;
  background: #f8f9fc !important;
  color: #1a1d26 !important;
  --body-background-fill: #f8f9fc !important;
  --body-text-color: #1a1d26 !important;
  --block-background-fill: #ffffff !important;
  --block-label-background-fill: #ffffff !important;
  --block-label-text-color: #64748b !important;
  --input-background-fill: #ffffff !important;
  --input-text-color: #1a1d26 !important;
  --border-color-primary: #e8ecf1 !important;
  --border-color-accent: #d1d9e6 !important;
  font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif !important;
}
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container p, .gradio-container label, .gradio-container span { color: #1a1d26; }
.gradio-container .main, .gradio-container .wrap { width: 100% !important; max-width: none !important; }

/* ── Page shell ── */
.page-shell {
  width: 100% !important;
  max-width: 1120px !important;
  margin: 0 auto;
}

/* ── Hero ── */
.hero {
  padding: 2.5rem 0 1.5rem;
  text-align: center;
}
.hero h1 {
  max-width: 700px;
  margin: 0 auto .6rem;
  font-size: clamp(1.75rem, 4.5vw, 2.5rem);
  line-height: 1.15;
  letter-spacing: -.035em;
  font-weight: 800;
  color: #111827 !important;
}
.hero p {
  max-width: 520px;
  margin: 0 auto;
  color: #6b7280 !important;
  font-size: 1rem;
  line-height: 1.65;
  font-weight: 400;
}

/* ── Eyebrow badge ── */
.eyebrow {
  display: inline-block;
  margin-bottom: .85rem !important;
  padding: .3rem .9rem;
  background: linear-gradient(135deg, #eef2ff, #e0e7ff);
  border: 1px solid #c7d2fe;
  border-radius: 100px;
  color: #4338ca !important;
  font-size: .72rem !important;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
}

/* ── Card panels ── */
.panel {
  min-width: 0 !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 20px !important;
  padding: 1.5rem !important;
  background: #ffffff !important;
  box-shadow:
    0 1px 2px rgba(0,0,0,.04),
    0 4px 16px rgba(0,0,0,.03) !important;
  transition: box-shadow .25s ease, border-color .25s ease;
}
.panel:hover {
  border-color: #d1d5db !important;
  box-shadow:
    0 1px 2px rgba(0,0,0,.04),
    0 8px 32px rgba(0,0,0,.06) !important;
}
.panel, .panel * { color: #111827; }

/* ── Panel section title ── */
.section-title h3 {
  margin: 0 0 .15rem !important;
  font-size: .95rem !important;
  font-weight: 700 !important;
  color: #111827 !important;
  letter-spacing: -.01em;
}
.section-title p {
  margin: 0 0 .65rem !important;
  color: #9ca3af !important;
  font-size: .82rem !important;
  font-weight: 400;
}

/* ── Image upload area ── */
.panel .image-container,
.panel [data-testid="image"] {
  border-radius: 14px !important;
  overflow: hidden !important;
}
.panel .upload-container,
.panel [data-testid="image"] > div:first-child {
  border: 2px dashed #d1d5db !important;
  border-radius: 14px !important;
  background: #fafbfd !important;
  transition: border-color .2s ease, background .2s ease;
}
.panel .upload-container:hover,
.panel [data-testid="image"] > div:first-child:hover {
  border-color: #818cf8 !important;
  background: #f5f3ff !important;
}

/* ── Textarea output ── */
.panel textarea {
  border: 1px solid #e5e7eb !important;
  border-radius: 14px !important;
  background: #fafbfd !important;
  color: #111827 !important;
  font-size: 1rem !important;
  line-height: 1.7 !important;
  padding: .85rem 1rem !important;
  transition: border-color .2s ease;
}
.panel textarea:focus {
  border-color: #818cf8 !important;
  box-shadow: 0 0 0 3px rgba(129,140,248,.12) !important;
}
.panel button { border-radius: 12px !important; }

/* ── Action row buttons ── */
.action-row {
  margin-top: 1rem;
  gap: .75rem !important;
}
.action-row button {
  min-height: 48px;
  border-radius: 14px !important;
  font-weight: 650 !important;
  font-size: .92rem !important;
  letter-spacing: -.005em;
  transition: all .2s ease !important;
}
/* Primary button */
.action-row button.primary {
  background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
  border: none !important;
  color: #fff !important;
  box-shadow: 0 2px 8px rgba(79,70,229,.25) !important;
}
.action-row button.primary:hover {
  background: linear-gradient(135deg, #4338ca, #4f46e5) !important;
  box-shadow: 0 4px 16px rgba(79,70,229,.35) !important;
  transform: translateY(-1px);
}
.action-row button.primary:active {
  transform: translateY(0);
  box-shadow: 0 1px 4px rgba(79,70,229,.2) !important;
}
/* Secondary / clear button */
.action-row button.secondary,
.action-row button:not(.primary) {
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  color: #374151 !important;
}
.action-row button.secondary:hover,
.action-row button:not(.primary):hover {
  background: #f9fafb !important;
  border-color: #d1d5db !important;
}

/* ── Helper text ── */
.helper {
  margin: 1.15rem 0 0 !important;
  color: #9ca3af !important;
  text-align: center;
  font-size: .82rem !important;
  font-weight: 400;
}

/* ── Footer ── */
.site-footer {
  margin-top: 2rem !important;
  padding: 1.15rem 0 1.5rem !important;
  border-top: 1px solid #e5e7eb;
  color: #9ca3af !important;
  text-align: center;
  font-size: .78rem !important;
  letter-spacing: .01em;
}
.site-footer strong {
  color: #374151 !important;
  font-weight: 700;
}

/* ── Hide Gradio branding ── */
.site-footer + footer, footer,
a[href*="gradio.app"],
button[aria-label*="Settings"],
button[aria-label*="Use via API"] { display: none !important; }

/* ── Responsive ── */
@media (max-width: 768px) {
  .gradio-container { padding: 0 16px !important; }
  .hero { padding: 1.75rem 0 1rem; }
  .hero h1 { font-size: 1.6rem; }
  .panel { padding: 1rem !important; border-radius: 16px !important; }
  .action-row button { min-height: 44px; border-radius: 12px !important; }
}

/* ── Smooth global transitions ── */
*, *::before, *::after {
  transition-property: color, background-color, border-color, box-shadow;
  transition-duration: .15s;
  transition-timing-function: ease;
}
"""


with gr.Blocks(
    title="Image Caption Generation",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css=CSS,
) as demo:
    with gr.Column(elem_classes=["page-shell"]):
        gr.Markdown(
            "<p class='eyebrow'>✦ AI-Powered Vision</p>"
            "<h1>Image Caption Generation</h1>"
            "<p>Upload any photo and let AI describe it in natural language — instantly.</p>",
            elem_classes=["hero"],
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes=["panel"]):
                gr.Markdown(
                    "### 📷 Upload Image\n"
                    "Drag & drop or click to choose a photo.",
                    elem_classes=["section-title"],
                )
                image_input = gr.Image(
                    label="",
                    type="pil",
                    image_mode="RGB",
                    sources=["upload"],
                    height=340,
                    show_label=False,
                )
            with gr.Column(scale=1, elem_classes=["panel"]):
                gr.Markdown(
                    "### 💬 Generated Caption\n"
                    "Your AI-generated description appears below.",
                    elem_classes=["section-title"],
                )
                caption_output = gr.Textbox(
                    label="",
                    lines=9,
                    interactive=False,
                    show_label=False,
                    placeholder="Your caption will appear here once generated…",
                )
        with gr.Row(elem_classes=["action-row"]):
            describe_button = gr.Button(
                "✨ Generate Caption", variant="primary", size="lg",
            )
            clear_button = gr.ClearButton(
                [image_input, caption_output], value="↺ Clear",
            )
        gr.Markdown(
            "💡 For best results, upload a clear image with the main subject visible.",
            elem_classes=["helper"],
        )
        gr.Markdown(
            "<strong>Prasad V Potluri Siddhartha Institute of Technology</strong>",
            elem_classes=["site-footer"],
        )
    describe_button.click(generate_caption, inputs=image_input, outputs=caption_output)


if __name__ == "__main__":
    demo.queue().launch(show_api=False, width="100%")
