

import gradio as gr
import datetime as DT
import pytz
from gradio_client import Client

ipAddress = None


def __nowInIST():
    return DT.datetime.now(pytz.timezone("Asia/Kolkata"))


def __attachIp(request: gr.Request):
    global ipAddress
    x_forwarded_for = request.headers.get('x-forwarded-for')
    if x_forwarded_for:
        ipAddress = x_forwarded_for


def pprint(log: str):
    now = __nowInIST()
    now = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [{ipAddress}] {log}")


def predict(audio):
    pprint("Starting the job")
    client = Client("https://abidlabs-music-separation.hf.space/")
    result = client.predict(
        audio,
        api_name="/predict"
    )
    pprint(f"{result=}")
    return result


with gr.Interface(
    predict,
    inputs=gr.Audio(type="filepath", label="Input"),
    outputs=[
      gr.Audio(type="filepath", label="Vocals"),
      gr.Audio(type="filepath", label="No Vocals / Instrumental")
    ],
    title="Split your song into vocals & music",
    article="<p style='text-align: center'>Credits: <a href='https://huggingface.co/spaces/abidlabs/music-separation'>abidlabs/music-separation</a> </>",
) as demo:
  demo.load(__attachIp, None, None)

demo.launch(debug=True)

