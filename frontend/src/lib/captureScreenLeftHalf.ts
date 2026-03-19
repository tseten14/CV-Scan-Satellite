// Capture a screen/window via getDisplayMedia, then crop the left 50% of the frame.
// Best when the user shares the browser window: left pane ≈ map.

export async function captureScreenLeftHalfAsPngFile(): Promise<File> {
  if (!navigator.mediaDevices?.getDisplayMedia) {
    throw new Error("Screen capture is not supported in this browser.");
  }

  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: false,
  });

  const video = document.createElement("video");
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;

  try {
    await video.play();

    await new Promise<void>((resolve) => {
      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        resolve();
        return;
      }
      video.addEventListener("loadeddata", () => resolve(), { once: true });
    });

    await new Promise<void>((r) => requestAnimationFrame(() => r()));

    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) {
      throw new Error("Could not read capture size.");
    }

    const cropW = Math.floor(w / 2);
    const canvas = document.createElement("canvas");
    canvas.width = cropW;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Could not create canvas.");
    }

    ctx.drawImage(video, 0, 0, cropW, h, 0, 0, cropW, h);

    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(
        (b) => (b ? resolve(b) : reject(new Error("Failed to encode image"))),
        "image/png"
      );
    });

    return new File([blob], "screen-left-half.png", { type: "image/png" });
  } finally {
    stream.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }
}
