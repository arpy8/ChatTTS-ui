# ChatTTS WebUI & API 

A simple local web interface to use [ChatTTS](https://github.com/2noise/chattts) for text-to-speech synthesis directly on a webpage, and it also supports API integration.

[Download the Windows integrated package from the Releases](https://github.com/jianchang512/ChatTTS-ui/releases).

> Interface Preview
>
> ![image](https://github.com/jianchang512/ChatTTS-ui/assets/3378335/6ed7c993-3882-4c34-9abd-f0635b133012)
>

Listen to the synthesized speech sample

https://github.com/jianchang512/ChatTTS-ui/assets/3378335/03cf1c0f-0245-44b5-8007-370d9db2bda8

## Pre-packaged Version for Windows

1. Download the compressed package from the [Releases](https://github.com/jianchang512/chatTTS-ui/releases), unzip it, and double-click app.exe to use it.

## Source Deployment on Linux

1. Set up a python3.9+ environment.
2. Create an empty directory `/data/chattts`, execute the command `cd /data/chattts && git clone https://github.com/jianchang512/chatTTS-ui .`.
3. Create a virtual environment `python3 -m venv venv`.
4. Activate the virtual environment `source ./venv/bin/activate`.
5. Install dependencies `pip3 install -r requirements.txt`.
6. If you do not need CUDA acceleration, execute `pip3 install torch torchaudio`.

    If CUDA acceleration is needed, execute:
    ```
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
    ```
    Additionally, install CUDA11.8+ ToolKit by searching for installation methods or refer to https://juejin.cn/post/7318704408727519270.
    
7. Execute `python3 app.py` to start the application, which will automatically open a browser window. The default address is `http://127.0.0.1:9966`.

## Source Deployment on MacOS

1. Set up a python3.9+ environment and install git. Execute `brew install git python@3.10`.
    Continue executing:
    ```
    export PATH="/usr/local/opt/python@3.10/bin:$PATH"
    source ~/.bash_profile
    source ~/.zshrc
    ```
2. Create an empty directory `/data/chattts`, execute the command `cd /data/chattts && git clone https://github.com/jianchang512/chatTTS-ui .`.
3. Create a virtual environment `python3 -m venv venv`.
4. Activate the virtual environment `source ./venv/bin/activate`.
5. Install dependencies `pip3 install -r requirements.txt`.
6. Install torch `pip3 install torch torchaudio`.
7. Execute `python3 app.py` to start the application, which will automatically open a browser window. The default address is `http://127.0.0.1:9966`.
8. For issues on MacOS, please check the [FAQ](faq.md).

## Source Deployment on Windows

1. Download and install python3.9+, and ensure to select `Add Python to environment variables` during installation.
2. Download and install git from https://github.com/git-for-windows/git/releases/download/v2.45.1.windows.1/Git-2.45.1-64-bit.exe.
3. Create an empty folder `D:/chattts` and enter it. In the address bar, type `cmd` and press enter. In the opened cmd window, execute `git clone https://github.com/jianchang512/chatTTS-ui .`.
4. Create a virtual environment by executing `python -m venv venv`.
5. Activate the virtual environment by executing `.\venv\scripts\activate`.
6. Install dependencies by executing `pip install -r requirements.txt`.
7. If you do not need CUDA acceleration, execute `pip install torch torchaudio`.

    If CUDA acceleration is needed, execute:
    `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118`
    
    Additionally, install CUDA11.8+ ToolKit by searching for installation methods or refer to https://juejin.cn/post/7318704408727519270.
    
8. Execute `python app.py` to start the application, which will automatically open a browser window. The default address is `http://127.0.0.1:9966`.

## Notes for Source Deployment

1. After starting the source deployment, it will first download the model from modelscope. However, modelscope lacks `spk_stat.pt` and will report an error. Download `spk_stat.pt` from https://huggingface.co/2Noise/ChatTTS/blob/main/asset/spk_stat.pt and copy it to the `project directory/models/pzc163/chatTTS/asset/` folder.

2. Note that modelscope only allows model downloads from mainland China IP addresses. If you encounter proxy errors, disable the proxy. If you want to download the model from huggingface.co, open `app.py` and check the comments around lines 50-60.

3. If GPU acceleration is needed, you must have an NVIDIA graphics card and install the CUDA version of torch. `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118`.

```
# Download model from modelscope by default. To download from huggingface, comment out the following 3 lines:
CHATTTS_DIR = snapshot_download('pzc163/chatTTS',cache_dir=MODEL_DIR)
chat = ChatTTS.Chat()
chat.load_models(source="local",local_path=CHATTTS_DIR)

# To download from huggingface.co, uncomment the following lines and comment out the above 3 lines:
# os.environ['HF_HUB_CACHE']=MODEL_DIR
# os.environ['HF_ASSETS_CACHE']=MODEL_DIR
# chat = ChatTTS.Chat()
# chat.load_models()
```

## [FAQ](faq.md)

## Modify HTTP Address

The default address is `http://127.0.0.1:9966`. To modify it, open the `.env` file in the directory and change `WEB_ADDRESS=127.0.0.1:9966` to the appropriate IP and port, such as `WEB_ADDRESS=192.168.0.10:9966` for access within the local network.

## Using the API v0.5+

**Request Method:** POST

**Request URL:** http://127.0.0.1:9966/tts

**Request Parameters:**

- `text`: str | Required, text to synthesize.
- `voice`: int | Optional, default 2222, determines the voice tone, can be 2222, 7869, 6653, 4099, 5099, or any value for a random voice tone.
- `prompt`: str | Optional, default empty, set laughter, pauses, e.g., [oral_2][laugh_0][break_6].
- `temperature`: float | Optional, default 0.3.
- `top_p`: float | Optional, default 0.7.
- `top_k`: int | Optional, default 20.
- `skip_refine`: int | Optional, default 0, 1=skip refine text, 0=do not skip.
- `custom_voice`: int | Optional, default 0, seed value for custom voice tone, greater than 0. If set, it takes precedence over `voice`.

**Response: JSON Data**

Successful response:
```
{
  code: 0,
  msg: 'ok',
  audio_files: [dict1, dict2]
}
```
Where `audio_files` is an array of dictionaries, each element is a dictionary with `{filename: absolute path of the wav file, url: downloadable wav URL}`.

Failed response:
```
{
  code: 1,
  msg: "error reason"
}
```

```python
# API Call Code

import requests

res = requests.post('http://127.0.0.1:9966/tts', data={
  "text": "If unsure, leave it blank",
  "prompt": "",
  "voice": "3333",
  "temperature": 0.3,
  "top_p": 0.7,
  "top_k": 20,
  "skip_refine": 0,
  "custom_voice": 0
})
print(res.json())

#ok
{
  code: 0,
  msg: 'ok',
  audio_files: [{
    filename: "E:/python/chattts/static/wavs/20240601-22_12_12-c7456293f7b5e4dfd3ff83bbd884a23e.wav",
    url: "http://127.0.0.1:9966/static/wavs/20240601-22_12_12-c7456293f7b5e4dfd3ff83bbd884a23e.wav"
  }]
}

#error
{
  code: 1,
  msg: "error"
}
```

## Using in pyVideoTrans Software

> Upgrade pyVideoTrans to 1.82+ https://github.com/jianchang512/pyvideotrans

1. Click Menu - Settings - ChatTTS, fill in the request URL, which should be `http://127.0.0.1:9966` by default

.
2. After testing, select `ChatTTS` on the main interface.

![image](https://github.com/jianchang512/ChatTTS-ui/assets/3378335/7118325f-2b9a-46ce-a584-1d5c6dc8e2da)