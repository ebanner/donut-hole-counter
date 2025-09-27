from flask import Flask, send_file, make_response
from PIL import Image
import io

from ultralytics import YOLO
import cv2
from PIL import Image
from IPython.display import display


app = Flask(__name__)

model = YOLO("/Users/edward/Code/dunkin-munchkin-counter/runs/detect/train2/weights/best.pt")

@app.route("/image")
def get_image():
    results = model.predict(
        source='data/train/images/2dd1ddf2-IMG_1326.jpeg',
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

    return send_file(buf, mimetype="image/png")


@app.route("/")
def index():
    return """
    <!doctype html>
    <html>
      <body style="font-family: system-ui, -apple-system, sans-serif;">
        <button onclick="loadImage()">Count Munchkins</button>
        <div id="count-container" style="margin-top:12px; display:none;">
            <strong>Munchkins:</strong> <span id="count">–</span>
        </div>
        <div style="margin-top:12px">
          <img id="preview" style="max-width:100%; border:1px solid #ccc;" />
        </div>
        <script>
          async function loadImage() {
            const resp = await fetch("/image?" + Date.now());
            const count = resp.headers.get("X-Detections");
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);

            document.getElementById("count").textContent = count ?? "0";
            document.getElementById("count-container").style.display = "block";

            const img = document.getElementById("preview");
            // Revoke previous object URL if any
            if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
            img.src = url;
            img.dataset.url = url;
          }
        </script>
      </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)
