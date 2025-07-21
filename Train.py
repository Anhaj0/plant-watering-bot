import os, math, tensorflow as tf
from pathlib import Path

# ───────── CONFIG ─────────
DATA_DIR      = 'plants'        # 'plant' / 'not_plant' sub-folders
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 64
SEED          = 42
VAL_SPLIT     = 0.30
EPOCHS_PHASE  = 10              # each augmentation phase
EPOCHS_FINE   = 5               # fine-tune all layers
REP_SAMPLES   = 100             # for INT8 rep-dataset
MODEL_DIR     = Path('model');  MODEL_DIR.mkdir(exist_ok=True)

# ───────── AUGMENT HELPERS ─────────
_clip = lambda img: tf.clip_by_value(img, 0., 1.)

def _rand_zoom(img, zoom=0.7):
    h,w = IMG_SIZE
    return tf.image.resize(tf.image.central_crop(img, zoom), (h,w))

def _rand_rot(img):
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)  # 0,1,2,3 × 90°
    return tf.image.rot90(img, k)

def _noise(img, std=0.05):
    return _clip(img + tf.random.normal(tf.shape(img), 0., std))

# --- Phase-specific augmentations ---
def rgb_standard(img):
    img = _rand_rot(img); img = _rand_zoom(img)
    img = tf.image.random_flip_left_right(img)
    delta = tf.random.uniform([], -0.4, 0.4)          # 0.6–1.4 range
    img = tf.image.adjust_brightness(img, delta)
    return _noise(img)

def rgb_dark(img):
    img = _rand_rot(img)
    delta = tf.random.uniform([], -0.8, -0.2)         # dark boost
    img = tf.image.adjust_brightness(img, delta)
    return _noise(img)

def rgb_bright(img):
    img = _rand_rot(img)
    delta = tf.random.uniform([], 0.4, 0.8)           # bright boost
    img = tf.image.adjust_brightness(img, delta)
    return _noise(img)

def rot90(img):  return _noise(tf.image.rot90(img, 1))
def rot180(img): return _noise(tf.image.rot90(img, 2))
def rot270(img): return _noise(tf.image.rot90(img, 3))

PHASES_RGB = [
    ('rgb_standard', rgb_standard),
    ('rgb_dark',     rgb_dark),
    ('rgb_bright',   rgb_bright),
    ('rgb_rot90',    rot90),
    ('rgb_rot180',   rot180),
    ('rgb_rot270',   rot270),
]
PHASES_GRAY = [
    ('gray_standard', rgb_standard),
    ('gray_dark',     rgb_dark),
    ('gray_bright',   rgb_bright),
    ('gray_rot90',    rot90),
    ('gray_rot180',   rot180),
    ('gray_rot270',   rot270),
]

# ───────── UTILITIES ─────────
def load_and_resize(path):
    img = tf.image.decode_png(tf.io.read_file(path), channels=3)
    return tf.cast(tf.image.resize(img, IMG_SIZE), tf.uint8)

def resize_and_rename():
    for lbl in ['plant','not_plant']:
        folder = os.path.join(DATA_DIR,lbl)
        for i,f in enumerate(sorted(tf.io.gfile.listdir(folder)),1):
            p_old = os.path.join(folder,f);  p_new = os.path.join(folder,f'{i}.png')
            try:
                tf.io.write_file(p_new, tf.image.encode_png(load_and_resize(p_old)))
                if p_new!=p_old: tf.io.gfile.remove(p_old)
            except Exception as e: print('Err',p_old,e)

def get_ds():
    train = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR, validation_split=VAL_SPLIT, subset='training', seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    val   = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR, validation_split=VAL_SPLIT, subset='validation', seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    norm  = tf.keras.layers.Rescaling(1./255)
    train = train.map(lambda x,y:(norm(x),y)).prefetch(tf.data.AUTOTUNE)
    val   = val.map(lambda x,y:(norm(x),y)).prefetch(tf.data.AUTOTUNE)
    return train,val

# ───────── MODEL ─────────
def make_model():
    base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE+(3,),
                                             include_top=False, weights='imagenet')
    base.trainable=False
    inp = tf.keras.Input(shape=IMG_SIZE+(3,))
    x   = tf.keras.layers.GlobalAveragePooling2D()(base(inp, training=False))
    x   = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    m   = tf.keras.Model(inp,out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy', metrics=['accuracy'])
    return m

# ───────── TRAIN ONE PHASE ─────────
def run_phase(model, train,val,name,aug_fn):
    print(f'\n── Phase: {name} ──')
    ds = train.unbatch().map(lambda x,y:(aug_fn(x),y),
                             num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    model.fit(ds, validation_data=val, epochs=EPOCHS_PHASE)

# ───────── GRAYSCALE CONVERT ─────────
def to_gray():
    for lbl in ['plant','not_plant']:
        folder=os.path.join(DATA_DIR,lbl)
        for f in tf.io.gfile.listdir(folder):
            p=os.path.join(folder,f)
            img=tf.image.decode_png(tf.io.read_file(p),channels=3)
            gray=tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img))
            tf.io.write_file(p,tf.image.encode_png(tf.cast(gray,tf.uint8)))

# ───────── FINE-TUNE ALL LAYERS ─────────
def fine_tune(model,train,val):
    print('\n── Fine-tuning ALL layers ──')
    for l in model.layers[1].layers: l.trainable=True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(train, validation_data=val, epochs=EPOCHS_FINE)
    model.save(MODEL_DIR/'mobilenet_finetuned.h5')

# ───────── QUANTIZE ─────────
def to_int8(model,train):
    print('\n── INT8 export ──')
    def rep(): 
        for img,_ in train.unbatch().batch(1).take(REP_SAMPLES): yield [img]
    conv=tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations=[tf.lite.Optimize.DEFAULT]; conv.representative_dataset=rep
    conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type=tf.uint8; conv.inference_output_type=tf.uint8
    (MODEL_DIR/'plant_model_int8.tflite').write_bytes(conv.convert())
    print('Saved → model/plant_model_int8.tflite')

# ───────── MAIN ─────────
if __name__=='__main__':
    resize_and_rename()
    train,val = get_ds()
    model = make_model()

    for n,f in PHASES_RGB:  run_phase(model,train,val,n,f)

    to_gray();  train,val = get_ds()
    for n,f in PHASES_GRAY: run_phase(model,train,val,n,f)

    fine_tune(model,train,val)
    to_int8(model,train)
