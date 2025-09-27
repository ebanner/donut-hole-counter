from flask import Flask, request, send_file, make_response
from PIL import Image
import io
import cv2

from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load your trained model (adjust the path if needed)
model = YOLO("models/best.pt")


@app.route("/", methods=["GET"])
def index():
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Munchkin Counter</title>
      <style>
        :root {
          --pad: 16px;
        }
        * { box-sizing: border-box; }
        body {
          font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
          margin: 0;
          padding: var(--pad);
          line-height: 1.3;
          background: #fff;
          color: #111;
        }
        .container {
          max-width: 720px;
          margin: 0 auto;
        }
        h1 {
          font-size: 1.9rem;
          margin: 0 0 0.75rem 0;
          text-align: left;
        }
        form#upload-form {
          display: flex;
          gap: 0.75rem;
          align-items: center;
          flex-wrap: wrap;
          margin-bottom: 0.75rem;
        }
        input[type="file"] {
          font-size: 1rem;
          padding: 0.65rem;
          border: 1px solid #d0d0d0;
          border-radius: 8px;
          background: #fafafa;
          max-width: 100%;
        }
        button {
          font-size: 1rem;
          padding: 0.7rem 1.1rem;
          border-radius: 8px;
          border: 1px solid #0a62ff;
          background: #1367ff;
          color: #fff;
        }
        #count-container {
          margin: 0.5rem 0 0.75rem 0;
          font-weight: 700;
          font-size: 1.5rem;
        }
        #preview-wrap {
          margin-top: 0.5rem;
        }
        img#preview {
          max-width: 100%;
          height: auto;
          border-radius: 10px;
          border: 1px solid #e1e1e1;
          display: block;
        }
        /* Rotate wide images for mobile viewing */
        .rotate90 {
          transform: rotate(90deg);
          transform-origin: center center;
          /* Ensure it fits vertically after rotation */
          max-height: 85vh;
          width: auto;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Munchkin Counter</h1>

        <form id="upload-form">
          <input type="file" name="file" id="file" accept="image/*" required>
          <button type="submit">Upload</button>
        </form>

        <div id="count-container" style="display:none;">
          Munchkins: <span id="count">–</span>
        </div>

        <div id="preview-wrap" style="display:none;">
          <img id="preview" alt="Detection result">
        </div>
      </div>

      <script>
        const form = document.getElementById("upload-form");
        const fileInput = document.getElementById("file");
        const previewWrap = document.getElementById("preview-wrap");
        const preview = document.getElementById("preview");
        const countContainer = document.getElementById("count-container");
        const countSpan = document.getElementById("count");
        const ROTATE_AR_THRESHOLD = 1.3;

        form.addEventListener("submit", async (e) => {
          e.preventDefault();
          const fd = new FormData();
          if (!fileInput.files.length) return;
          fd.append("file", fileInput.files[0]);

          // ✅ fixed: post to /upload instead of /predict
          const res = await fetch("/upload", { method: "POST", body: fd });
          if (!res.ok) {
            alert("Upload failed");
            return;
          }

          // right now /upload returns an image file, not JSON
          // so just display it directly
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);

          preview.classList.remove("rotate90");
          preview.src = url;
          previewWrap.style.display = "block";

          // read count from response header
          const count = res.headers.get("X-Detections");
          if (count !== null) {
            countSpan.textContent = count;
            countContainer.style.display = "block";
          }

          preview.onload = () => {
            const ar = preview.naturalWidth / preview.naturalHeight;
            if (ar > ROTATE_AR_THRESHOLD) {
              preview.classList.add("rotate90");
            }
          };
        });
      </script>
    </body>
    </html>
    """


@app.route("/upload", methods=["POST"])
def upload():
    """
    Echo the uploaded image back as the response.
    """
    f = request.files.get("file")
    if not f:
        return "No file part", 400

    # 1) Read as PIL
    img = Image.open(f.stream)
    img.load()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")  # e.g., convert RGBA/CMYK -> RGB

    results = model.predict(
        img,
        conf=0.025,
        project="test_overfitting",
        name="training_image_test"
    )

    result = results[0]

    annot = result.plot(labels=False, conf=False)

    img = Image.fromarray(
        cv2.cvtColor(annot, cv2.COLOR_BGR2RGB)
    )

    if img.width > img.height:
        img = img.rotate(90, expand=True)

    result, = results

    num_detections = len(result.boxes)

    # img = Image.new("RGB", (200, 100), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # send file + custom header for the count
    resp = make_response(send_file(buf, mimetype="image/png"))
    resp.headers["X-Detections"] = str(num_detections)
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

