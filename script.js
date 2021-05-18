let model, streamvideo, streamctx, ctx, videoWidth, videoHeight, video, canvas, facecanvas, facectx, image;
let current = new Date();

const state = {
  backend: "wasm",
};

async function setupCamera() {

  video = document.getElementById("video");

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: "user" },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}


const renderPrediction = async () => {
  const returnTensors = false;
  const flipHorizontal = false;
  const annotateBoxes = false;
  const predictions = await model.estimateFaces(
    video,
    returnTensors,
    flipHorizontal,
    annotateBoxes
  );
 

  if (predictions.length > 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < predictions.length; i++) {
      if (returnTensors) {
        predictions[i].topLeft = predictions[i].topLeft.arraySync();
        predictions[i].bottomRight = predictions[i].bottomRight.arraySync();
        if (annotateBoxes) {
          predictions[i].landmarks = predictions[i].landmarks.arraySync();
        }
      }

      const start = predictions[i].topLeft;
      const end = predictions[i].bottomRight;
      const size = [end[0] - start[0], end[1] - start[1]];
 
      ctx.beginPath();
      ctx.lineWidth = "6";
      ctx.strokeStyle = "red";
      ctx.rect(640-end[0], start[1], size[0], size[1]);
      ctx.stroke();

      frame = video;
      streamctx.drawImage(video, 0, 0)
      
      facectx.drawImage( 
        streamcanvas, // source 
        end[0], start[1], // sx, sy,
        start[0]-end[0], end[1]-start[1], //sWidth, sHeight
        0, 0, // dx, dy
        48, 48, // dWidth, dHeight
      )

      await classify(facecanvas);

      if (annotateBoxes) {
        const landmarks = predictions[i].landmarks;

        ctx.fillStyle = "blue";
        for (let j = 0; j < landmarks.length; j++) {
          const x = landmarks[j][0];
          const y = landmarks[j][1];
          ctx.fillRect(x, y, 5, 5);
        }
      }
    }
  }

  // window.requestAnimationFrame(renderPrediction);
  video.requestVideoFrameCallback(renderPrediction)
};

const setupPage = async () => {
  image = document.getElementById('img');
  await setupCamera();
  video.play();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  facecanvas = document.getElementById("face")
  facecanvas.width =  48;
  facecanvas.height =  48;
  facectx = facecanvas.getContext("2d");

  streamcanvas = document.getElementById("videostream")
  streamcanvas.width = videoWidth;
  streamcanvas.height = videoHeight;
  streamctx = streamcanvas.getContext("2d");
  streamctx.fillStyle = "rgba(255, 0, 0, 0.5)";

  canvas = document.getElementById("output");
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(255, 0, 0, 0.5)";

  model = await blazeface.load();
  emotimodel = await load();

  video.requestVideoFrameCallback(renderPrediction);
};

let emotimodel;
async function load() {
  return await tf.loadLayersModel("xception/model.json");
}

async function classify(img) {
  
  const tensor = await tf.browser.fromPixels(img,1).expandDims(0)
  const offset = tf.scalar(127.5);
  // Normalize the image from [0, 255] to [-1, 1].
  const normalized = tensor.sub(offset).div(offset);
  const res =  await emotimodel.predict(normalized).array();
  document.getElementById("pred").innerHTML = res;
}

load();
setupPage();
