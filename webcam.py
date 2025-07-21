import cv2, time, threading, queue, numpy as np, tensorflow as tf

# ───────────────────── CONFIG ─────────────────────
MODEL_PATH  = r"model\plant_model_int8.tflite"
NUM_ZONES   = 4               # vertical strips
CONF_THRESH = 0.80
CAM_INDEX   = 0
CAP_WIDTH   = 640             # smaller = faster
CAP_HEIGHT  = 480
NUM_THREADS = 6               # match your logical cores
# ──────────────────────────────────────────────────

# ---------- TFLite Interpreter (pre-allocated) ----------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH,
                                  num_threads=NUM_THREADS)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
in_scale, in_zero = inp['quantization']
out_scale, out_zero = out['quantization']
_, H, W, C = inp['shape']             # e.g. (1,128,128,1)

# fix input tensor to  (NUM_ZONES, H, W, C)  *once*
interpreter.resize_tensor_input(inp['index'], [NUM_ZONES, H, W, C])
interpreter.allocate_tensors()        # only ONCE
inp  = interpreter.get_input_details()[0]
out  = interpreter.get_output_details()[0]

# pre-allocate numpy array we’ll fill every frame
tfl_buffer = np.empty((NUM_ZONES, H, W, C), dtype=inp['dtype'])

# ---------- Background inference thread ----------
def worker_loop(frame_q, result_q):
    while True:
        frame = frame_q.get()
        if frame is None:
            break

        h, w = frame.shape[:2]
        zone_w = w // NUM_ZONES

        # fill tfl_buffer in-place (no realloc)
        for i in range(NUM_ZONES):
            x1 = i*zone_w
            x2 = w if i==NUM_ZONES-1 else x1+zone_w
            roi = cv2.resize(frame[:, x1:x2], (W, H))

            if C == 1:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = roi[..., None]          # H,W,1
            else:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            if inp['dtype'] == np.uint8:      # quantised model
                tfl_buffer[i] = (roi / in_scale + in_zero).astype(np.uint8)
            else:                             # float model
                tfl_buffer[i] = roi.astype(np.float32) / 255.0

        interpreter.set_tensor(inp['index'], tfl_buffer)
        interpreter.invoke()
        raw = interpreter.get_tensor(out['index']).flatten()

        # dequantise + flip class
        probs = 1.0 - ((raw - out_zero) * out_scale)
        result_q.put(probs)

# queues for thread-safe hand-off
frame_q  = queue.Queue(maxsize=2)
result_q = queue.Queue(maxsize=2)
threading.Thread(target=worker_loop, args=(frame_q, result_q),
                 daemon=True).start()

# ---------- Webcam capture / display ----------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

zone_probs = [0.0]*NUM_ZONES
t0 = time.time(); frames = 0
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # send latest frame (drop if queue full)
    try:
        frame_q.put_nowait(frame)
    except queue.Full:
        pass

    # pull latest inference if available
    try:
        zone_probs = result_q.get_nowait()
    except queue.Empty:
        pass

    # draw UI
    h, w = frame.shape[:2]; zw = w // NUM_ZONES
    for i, p in enumerate(zone_probs):
        x1, x2 = i*zw, (w if i==NUM_ZONES-1 else (i+1)*zw)
        label  = "Plant" if p > CONF_THRESH else "Not Plant"
        color  = (0,255,0) if p > CONF_THRESH else (0,0,255)
        cv2.rectangle(frame,(x1,0),(x2,h),color,2)
        cv2.putText(frame,f"{label} {p*100:.1f}%",
                    (x1+5,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    # FPS counter
    frames += 1
    if frames == 30:
        fps = 30.0/(time.time()-t0); t0=time.time(); frames=0
    cv2.putText(frame,f"FPS {fps:.1f}", (10,h-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.imshow("30 infer/sec plant zones",frame)
    if cv2.waitKey(1) & 0xFF == 27: break

# clean up
frame_q.put(None)
cap.release()
cv2.destroyAllWindows()
