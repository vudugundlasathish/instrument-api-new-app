# # from fastapi import FastAPI, UploadFile, File
# # from keras.models import load_model

# # app = FastAPI()
# # model = None

# # @app.get("/")
# # async def read_root():
# #     return {"status": "running"}

# # @app.post("/predict/")
# # async def predict(file: UploadFile = File(...)):
# #     global model
# #     if model is None:
# #         model = load_model("app/instrument_model.h5")
# #     return {"result": "prediction logic here"}

# # import numpy as np
# # import librosa
# # from fastapi import FastAPI, UploadFile, File
# # from fastapi.responses import JSONResponse
# # from tensorflow.keras.models import load_model
# # import shutil
# # import soundfile as sf
# # import os
# # import logging

# # app = FastAPI()

# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # # Lazy load model
# # model = None
# # labels = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
# # full_labels = {
# #     "cel": "Cello",
# #     "cla": "Clarinet",
# #     "flu": "Flute",
# #     "gac": "Acoustic Guitar",
# #     "gel": "Electric Guitar",
# #     "org": "Organ",
# #     "pia": "Piano",
# #     "sax": "Saxophone",
# #     "tru": "Trumpet",
# #     "vio": "Violin",
# #     "voi": "Voice"
# # }

# # def get_model():
# #     global model
# #     if model is None:
# #         try:
# #             model = load_model("app/instrument_model.h5")
# #             logger.info("Model loaded successfully.")
# #         except Exception as e:
# #             logger.error(f"Error loading model: {e}")
# #             raise
# #     return model

# # def extract_mfcc(file_path, sr=22050, n_mfcc=13, max_len=1300):
# #     audio, _ = librosa.load(file_path, sr=sr, mono=True)
# #     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

# #     if mfcc.shape[1] < max_len:
# #         pad_width = max_len - mfcc.shape[1]
# #         mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
# #     else:
# #         mfcc = mfcc[:, :max_len]

# #     return mfcc

# # @app.post("/predict-instrument")
# # async def predict_instrument(file: UploadFile = File(...)):
# #     temp_filename = "temp.wav"
# #     try:
# #         with open(temp_filename, "wb") as buffer:
# #             shutil.copyfileobj(file.file, buffer)

# #         mfcc = extract_mfcc(temp_filename, n_mfcc=13, max_len=1300)
# #         mfcc_flat = mfcc.flatten().reshape(1, -1)

# #         model_instance = get_model()
# #         prediction = model_instance.predict(mfcc_flat)[0]

# #         threshold = 0.10
# #         filtered_probs = {
# #             full_labels[label]: float(score)
# #             for label, score in zip(labels, prediction)
# #             if score >= threshold
# #         }

# #         return JSONResponse({"instruments": filtered_probs})
# #     except Exception as e:
# #         logger.error(f"Error during prediction: {e}")
# #         return JSONResponse(status_code=500, content={"error": str(e)})
# #     finally:
# #         if os.path.exists(temp_filename):
# #             os.remove(temp_filename)

# # @app.get("/")
# # def read_root():
# #     return {"message": "Instrument classifier is running 🚀"}
# import os
# import shutil
# import logging
# import numpy as np
# import librosa
# import soundfile as sf
# import gc
# import psutil

# from fastapi import FastAPI, UploadFile, File, Request, HTTPException
# from fastapi.responses import JSONResponse
# from tensorflow.keras.models import load_model
# from tensorflow.keras import backend as K

# app = FastAPI()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# model = None
# labels = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
# full_labels = {
#     "cel": "Cello", "cla": "Clarinet", "flu": "Flute", "gac": "Acoustic Guitar",
#     "gel": "Electric Guitar", "org": "Organ", "pia": "Piano", "sax": "Saxophone",
#     "tru": "Trumpet", "vio": "Violin", "voi": "Voice"
# }

# def get_model():
#     global model
#     if model is None:
#         try:
#             model = load_model("app/instrument_model.h5")
#             logger.info("✅ Model loaded successfully.")
#         except Exception as e:
#             logger.error(f"❌ Error loading model: {e}")
#             raise
#     return model

# def extract_mfcc(file_path, sr=22050, n_mfcc=13, max_len=1300):
#     audio, _ = librosa.load(file_path, sr=sr, mono=True)
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

#     if mfcc.shape[1] < max_len:
#         pad_width = max_len - mfcc.shape[1]
#         mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
#     else:
#         mfcc = mfcc[:, :max_len]

#     return mfcc

# def log_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info().rss / 1024 / 1024
#     logger.info(f"📊 Memory usage: {mem:.2f} MB")

# @app.post("/predict-instrument")
# async def predict_instrument(request: Request, file: UploadFile = File(...)):
#     MAX_SIZE_MB = 10
#     content_length = request.headers.get("content-length")
#     if content_length and int(content_length) > MAX_SIZE_MB * 1024 * 1024:
#         raise HTTPException(status_code=413, detail="Audio file too large (max 10MB)")

#     temp_filename = "temp.wav"
#     try:
#         with open(temp_filename, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         log_memory_usage()

#         mfcc = extract_mfcc(temp_filename)
#         mfcc_flat = mfcc.flatten().reshape(1, -1)

#         model_instance = get_model()
#         prediction = model_instance.predict(mfcc_flat)[0]

#         threshold = 0.10
#         filtered_probs = {
#             full_labels[label]: float(score)
#             for label, score in zip(labels, prediction)
#             if score >= threshold
#         }

#         return JSONResponse({"instruments": filtered_probs})

#     except Exception as e:
#         logger.error(f"🚨 Prediction error: {e}")
#         return JSONResponse(status_code=500, content={"error": str(e)})
#     finally:
#         if os.path.exists(temp_filename):
#             os.remove(temp_filename)
#         # Clean up memory
#         K.clear_session()
#         gc.collect()
#         log_memory_usage()

# @app.get("/")
# def read_root():
#     return {"message": "🎶 Instrument classifier is running"}

# import os
# import shutil
# import logging
# import numpy as np
# import librosa
# import soundfile as sf
# import gc
# import psutil

# from fastapi import FastAPI, UploadFile, File, Request, HTTPException
# from fastapi.responses import JSONResponse
# from tensorflow.keras.models import load_model
# from tensorflow.keras import backend as K

# app = FastAPI()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# labels = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
# full_labels = {
#     "cel": "Cello", "cla": "Clarinet", "flu": "Flute", "gac": "Acoustic Guitar",
#     "gel": "Electric Guitar", "org": "Organ", "pia": "Piano", "sax": "Saxophone",
#     "tru": "Trumpet", "vio": "Violin", "voi": "Voice"
# }

# def extract_mfcc(file_path, sr=22050, n_mfcc=13, max_len=1300):
#     audio, _ = librosa.load(file_path, sr=sr, mono=True)
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

#     if mfcc.shape[1] < max_len:
#         pad_width = max_len - mfcc.shape[1]
#         mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
#     else:
#         mfcc = mfcc[:, :max_len]

#     return mfcc

# def log_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info().rss / 1024 / 1024
#     logger.info(f"📊 Memory usage: {mem:.2f} MB")

# @app.post("/predict-instrument")
# async def predict_instrument(request: Request, file: UploadFile = File(...)):
#     MAX_SIZE_MB = 10
#     content_length = request.headers.get("content-length")
#     if content_length and int(content_length) > MAX_SIZE_MB * 1024 * 1024:
#         raise HTTPException(status_code=413, detail="Audio file too large (max 10MB)")

#     temp_filename = "temp.wav"
#     try:
#         with open(temp_filename, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         log_memory_usage()

#         mfcc = extract_mfcc(temp_filename)
#         mfcc_flat = mfcc.flatten().reshape(1, -1)

#         # Load model for this request only
#         model_instance = load_model("app/instrument_model.h5")
#         logger.info("✅ Model loaded successfully.")

#         prediction = model_instance.predict(mfcc_flat)[0]

#         threshold = 0.10
#         filtered_probs = {
#             full_labels[label]: float(score)
#             for label, score in zip(labels, prediction)
#             if score >= threshold
#         }

#         return JSONResponse({"instruments": filtered_probs})

#     except Exception as e:
#         logger.error(f"🚨 Prediction error: {e}")
#         return JSONResponse(status_code=500, content={"error": str(e)})
#     finally:
#         if os.path.exists(temp_filename):
#             os.remove(temp_filename)
#         # Clean up memory
#         K.clear_session()
#         gc.collect()
#         log_memory_usage()

# @app.get("/")
# def read_root():
#     return {"message": "🎶 Instrument classifier is running"}
import os
import shutil
import logging
import numpy as np
import librosa
import soundfile as sf
import gc
import psutil
import threading

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

labels = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
full_labels = {
    "cel": "Cello", "cla": "Clarinet", "flu": "Flute", "gac": "Acoustic Guitar",
    "gel": "Electric Guitar", "org": "Organ", "pia": "Piano", "sax": "Saxophone",
    "tru": "Trumpet", "vio": "Violin", "voi": "Voice"
}

# Singleton model loader with threading lock
class ModelWrapper:
    _model = None
    _lock = threading.Lock()

    @classmethod
    def get_model(cls):
        with cls._lock:
            if cls._model is None:
                cls._model = load_model("app/instrument_model.h5")
                logger.info("✅ Model loaded once and cached.")
            return cls._model

def extract_mfcc(file_path, sr=22050, n_mfcc=13, max_len=1300):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    logger.info(f"📊 Memory usage: {mem:.2f} MB")

def cleanup_resources():
    K.clear_session()
    gc.collect()
    log_memory_usage()

@app.post("/predict-instrument")
async def predict_instrument(request: Request, file: UploadFile = File(...)):
    MAX_SIZE_MB = 10
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large (max 10MB)")

    temp_filename = "temp.wav"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        log_memory_usage()

        # Optional: Move this to background in future
        mfcc = extract_mfcc(temp_filename)
        mfcc_flat = mfcc.flatten().reshape(1, -1)

        model_instance = ModelWrapper.get_model()
        prediction = model_instance.predict(mfcc_flat)[0]

        threshold = 0.10
        filtered_probs = {
            full_labels[label]: float(score)
            for label, score in zip(labels, prediction)
            if score >= threshold
        }

        return JSONResponse({"instruments": filtered_probs})

    except Exception as e:
        logger.error(f"🚨 Prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        cleanup_resources()

@app.get("/")
def read_root():
    return {"message": "🎶 Instrument classifier is running"}