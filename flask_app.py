from flask import Flask, request, send_file, make_response
from PIL import Image
import io
import cv2

from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load your trained model (adjust the path if needed)
model = YOLO("/Users/edward/Code/dunkin-munchkin-counter/runs/detect/train2/weights/best.pt")


@app.route("/", methods=["GET"])
def index():
    return """
    <!doctype html>
    <html>
      <body style="font-family: system-ui, -apple-system, sans-serif; padding: 16px;">
        <h1>Munchkin Counter</h1>

        <form id="upload-form">
          <input type="file" name="file" id="file" accept="image/*" required />
          <button type="submit">Upload</button>
        </form>

        <div id="count-container" style="margin-top:30px; display:none;">
          <strong>Munchkins:</strong> <span id="count">–</span>
        </div>

        <div id="preview-wrap" style="margin-top:16px; display:none;">
          <img id="preview" style="max-width:100%; border:1px solid #ccc;" />
        </div>

        <script>
          const form = document.getElementById("upload-form");
          const previewWrap = document.getElementById("preview-wrap");
          const preview = document.getElementById("preview");
          const countContainer = document.getElementById("count-container");
          const countSpan = document.getElementById("count");

          form.addEventListener("submit", async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById("file");
            if (!fileInput.files.length) return;

            const fd = new FormData();
            fd.append("file", fileInput.files[0]);

            const resp = await fetch("/upload", { method: "POST", body: fd });
            if (!resp.ok) {
              alert("Upload failed");
              return;
            }

            // Get detection count from header
            const dets = resp.headers.get("X-Detections");

            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);

            if (preview.dataset.url) URL.revokeObjectURL(preview.dataset.url);
            preview.src = url;
            preview.dataset.url = url;

            // Show image + count
            previewWrap.style.display = "block";
            countContainer.style.display = "block";
            countSpan.textContent = dets ?? "0";
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
        conf=0.5,
        project="test_overfitting",
        name="training_image_test"
    )

    result = results[0]

    annot = result.plot(labels=False, conf=False)

    img = Image.fromarray(
        cv2.cvtColor(annot, cv2.COLOR_BGR2RGB)
    )

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
    app.run(debug=True)

