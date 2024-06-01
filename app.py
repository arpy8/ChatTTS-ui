import os
import sys
from pathlib import Path
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
VERSION='0.6'

def get_executable_path():
    # This function returns the directory where the executable file is located
    if getattr(sys, 'frozen', False):
        # If the program is "frozen" and packaged, use this path
        return Path(sys.executable).parent.as_posix()
    else:
        return Path.cwd().as_posix()

ROOT_DIR=get_executable_path()

MODEL_DIR_PATH=Path(ROOT_DIR+"/models")
MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)
MODEL_DIR=MODEL_DIR_PATH.as_posix()

WAVS_DIR_PATH=Path(ROOT_DIR+"/static/wavs")
WAVS_DIR_PATH.mkdir(parents=True, exist_ok=True)
WAVS_DIR=WAVS_DIR_PATH.as_posix()

LOGS_DIR_PATH=Path(ROOT_DIR+"/logs")
LOGS_DIR_PATH.mkdir(parents=True, exist_ok=True)
LOGS_DIR=LOGS_DIR_PATH.as_posix()

import soundfile as sf
import ChatTTS
import datetime
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify,  send_from_directory
import logging
from logging.handlers import RotatingFileHandler
from waitress import serve
load_dotenv()
import hashlib, webbrowser
from modelscope import snapshot_download
import numpy as np
import time
# Read .env variables
WEB_ADDRESS = os.getenv('WEB_ADDRESS', '127.0.0.1:9966')

# By default, download the model from modelscope. If you want to download the model from huggingface, comment out the following 3 lines
CHATTTS_DIR = snapshot_download('pzc163/chatTTS',cache_dir=MODEL_DIR)
chat = ChatTTS.Chat()
chat.load_models(source="local",local_path=CHATTTS_DIR)

# If you want to download the model from huggingface.co, uncomment the following lines and comment out the above 3 lines
#os.environ['HF_HUB_CACHE']=MODEL_DIR
#os.environ['HF_ASSETS_CACHE']=MODEL_DIR
#chat = ChatTTS.Chat()
#chat.load_models()

# Configure logging
# Disable Werkzeug's default logging handler
log = logging.getLogger('werkzeug')
log.handlers[:] = []
log.setLevel(logging.WARNING)

app = Flask(__name__, 
    static_folder=ROOT_DIR+'/static', 
    static_url_path='/static',
    template_folder=ROOT_DIR+'/templates')

root_log = logging.getLogger()  # Flask's root logger
root_log.handlers = []
root_log.setLevel(logging.WARNING)
app.logger.setLevel(logging.WARNING) 
# Create a RotatingFileHandler object, set the file path and size limit
file_handler = RotatingFileHandler(LOGS_DIR+f'/{datetime.datetime.now().strftime("%Y%m%d")}.log', maxBytes=1024 * 1024, backupCount=5)
# Create a log format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Set the level and format of the file handler
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)
# Add the file handler to the logger
app.logger.addHandler(file_handler)
app.jinja_env.globals.update(enumerate=enumerate)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)


@app.route('/')
def index():
    return render_template("index.html", weburl=WEB_ADDRESS, version=VERSION)


# Returns TTS results based on the text, returning filename=file name, url=download address
# The requester can choose which one to use as needed
# params:
#
# text: text to be synthesized
# voice: voice
# custom_voice: custom voice value
# skip_refine: 1=skip refine_text stage, 0=do not skip
# temperature
# top_p
# top_k
# prompt:
@app.route('/tts', methods=['GET', 'POST'])
def tts():
    # Original string
    text = request.args.get("text", "").strip() or request.form.get("text", "").strip()
    prompt = request.form.get("prompt", '')
    try:
        custom_voice = int(request.form.get("custom_voice", 0))
        voice = custom_voice if custom_voice > 0 else int(request.form.get("voice", 2222))
    except Exception:
        voice = 2222
    print(f'{voice=}, {custom_voice=}')
    temperature = float(request.form.get("temperature", 0.3))
    top_p = float(request.form.get("top_p", 0.7))
    top_k = int(request.form.get("top_k", 20))

    skip_refine = request.form.get("skip_refine", '0')
    
    app.logger.info(f"[tts]{text=}\n{voice=},{skip_refine=}\n")
    if not text:
        return jsonify({"code": 1, "msg": "text params lost"})
    std, mean = torch.load(f'{CHATTTS_DIR}/asset/spk_stat.pt').chunk(2)
    torch.manual_seed(voice)

    rand_spk = chat.sample_random_speaker()
    #rand_spk = torch.randn(768) * std + mean

    audio_files = []
    md5_hash = hashlib.md5()
    md5_hash.update(f"{text}-{voice}-{skip_refine}-{prompt}".encode('utf-8'))
    datename = datetime.datetime.now().strftime('%Y%m%d-%H_%M_%S')
    filename = datename + '-' + md5_hash.hexdigest() + ".wav"

    start_time = time.time()

    wavs = chat.infer([t for t in text.split("\n") if t.strip()], use_decoder=True, skip_refine_text=True if int(skip_refine) == 1 else False, params_infer_code={
        'spk_emb': rand_spk,
        'temperature': temperature,
        'top_P': top_p,
        'top_K': top_k
    }, params_refine_text={'prompt': prompt}, do_text_normalization=False)

    end_time = time.time()
    inference_time = end_time - start_time
    inference_time_rounded = round(inference_time, 2)
    print(f"Inference time: {inference_time_rounded} seconds")

    # Initialize an empty numpy array for later merging
    combined_wavdata = np.array([], dtype=wavs[0][0].dtype)  # Ensure dtype matches your wav data type

    for wavdata in wavs:
        combined_wavdata = np.concatenate((combined_wavdata, wavdata[0]))

    sample_rate = 24000  # Assuming 24kHz sample rate
    audio_duration = len(combined_wavdata) / sample_rate
    audio_duration_rounded = round(audio_duration, 2)
    print(f"Audio duration: {audio_duration_rounded} seconds")

    sf.write(WAVS_DIR + '/' + filename, combined_wavdata, 24000)

    audio_files.append({
        "filename": WAVS_DIR + '/' + filename,
        "url": f"http://{request.host}/static/wavs/{filename}",
        "inference_time": inference_time_rounded,
        "audio_duration": audio_duration_rounded
    })
    result_dict = {"code": 0, "msg": "ok", "audio_files": audio_files}
    # Compatible with pyVideoTrans interface call
    if len(audio_files) == 1:
        result_dict["filename"] = audio_files[0]['filename']
        result_dict["url"] = audio_files[0]['url']

    return jsonify(result_dict)

def ClearWav(directory):
    # Get all files and directories in the ../static/wavs directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    if not files:
        return False, "No wav files in wavs directory"

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                print(f"Skipped directory: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}, error message: {e}")
            return False, str(e)
    return True, "All wav files have been deleted."


@app.route('/clear_wavs', methods=['POST'])
def clear_wavs():
    dir_path = 'static/wavs'  # Directory where wav audio files are stored
    success, message = ClearWav(dir_path)
    if success:
        return jsonify({"code": 0, "msg": message})
    else:
        return jsonify({"code": 1, "msg": message})

try:
    host = WEB_ADDRESS.split(':')
    print(f'Starting: {host}')
    webbrowser.open(f'http://{WEB_ADDRESS}')
    serve(app, host=host[0], port=int(host[1]))
except Exception:
    pass