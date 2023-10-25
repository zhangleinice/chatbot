# import os
# from transformers import AutoProcessor, AutoModel
# from IPython.display import Audio
# import torchaudio
# import torch
# import scipy


# bark_processor = AutoProcessor.from_pretrained("models/bark-small")
# bark_model = AutoModel.from_pretrained("models/bark-small")

# # 中文不太好
# def play_voice():

#     inputs = bark_processor(
#         text=["今天天气真好"],
#         return_tensors="pt",
#     )

#     speech_values = bark_model.generate(**inputs, do_sample=True)

#     sampling_rate = bark_model.generation_config.sample_rate

#     return speech_values, sampling_rate

# speech_values, sampling_rate = play_voice()


# # 保存
# scipy.io.wavfile.write("data/bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())



# 可能python版本或者包有冲突
# from paddlespeech.cli.tts.infer import TTSExecutor

# tts_executor = TTSExecutor()

# text = "今天天气十分不错，百度也能做语音合成。"
# # 默认只支持中文
# # text = "今天天气十分不错，Paddle Speech也能做语音合成。"

# output_file = "data/paddlespeech1.wav"

# tts_executor(text=text, output=output_file)