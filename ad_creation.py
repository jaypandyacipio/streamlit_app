from openai import AzureOpenAI
import pandas as pd
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, TextClip, CompositeVideoClip, concatenate_audioclips
import json
import re
import cv2
import numpy as np
import moviepy.config as mp_config
from video_utils import convert_with_cropping, convert_with_padding, convert_with_blur, VideoConverterWithBlur

# from tts import generate_tts_for_segments

# import noisereduce as nr
# from scipy.io import wavfile
# import numpy as np

import time
import json
from azure.cognitiveservices.speech import AudioDataStream, SpeechSynthesizer, SpeechConfig, SpeechSynthesisOutputFormat, AudioConfig
from moviepy.editor import AudioFileClip, concatenate_audioclips

def combine_audio_files(input_files, output_file):
    # Load all audio files into a list of AudioFileClip objects
    audio_clips = [AudioFileClip(file) for file in input_files]

    # Concatenate all audio clips into one
    combined_audio = concatenate_audioclips(audio_clips)

    # Write the result to the output file
    combined_audio.write_audiofile(output_file)

# Azure TTS subscription key and region
subscription_key = 'f74fe7fc6f6a48879e486a5b33e1653d'
region = 'westus'

# Configure the speech synthesis
speech_config = SpeechConfig(subscription=subscription_key, region=region)
speech_config.set_speech_synthesis_output_format(SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)

# Voice and supported styles dictionary
voices_and_styles = {
    'de-DE-ConradNeural': ['cheerful'],
    'en-GB-RyanNeural': ['chat', 'cheerful'],
    'en-GB-SoniaNeural': ['cheerful', 'sad'],
    'en-IN-NeerjaNeural': ['cheerful', 'empathetic', 'newscast'],
    'en-US-AriaNeural': ['angry', 'chat', 'cheerful', 'customerservice', 'empathetic', 'excited', 'friendly', 'hopeful',
                         'narration-professional', 'newscast-casual', 'newscast-formal', 'sad', 'shouting', 'terrified',
                         'unfriendly', 'whispering'],
    'en-US-DavisNeural': ['angry', 'chat', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
                          'unfriendly', 'whispering'],
    'en-US-GuyNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'newscast', 'sad', 'shouting',
                        'terrified', 'unfriendly', 'whispering'],
    'en-US-JaneNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
                         'unfriendly', 'whispering'],
    'en-US-JasonNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
                          'unfriendly', 'whispering'],
    'en-US-JennyNeural': ['angry', 'assistant', 'chat', 'cheerful', 'customerservice', 'excited', 'friendly', 'hopeful',
                          'newscast', 'sad', 'shouting', 'terrified', 'unfriendly', 'whispering'],
    'en-US-KaiNeural': ['conversation'],
    'en-US-LunaNeural': ['conversation'],
}

from openai import AzureOpenAI
import pandas as pd
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, TextClip, CompositeVideoClip, concatenate_audioclips
import json
import re
import cv2
import numpy as np
import moviepy.config as mp_config
from video_utils import convert_with_cropping, convert_with_padding, convert_with_blur, VideoConverterWithBlur

# from tts import generate_tts_for_segments

# import noisereduce as nr
# from scipy.io import wavfile
# import numpy as np

import time
import json
from azure.cognitiveservices.speech import AudioDataStream, SpeechSynthesizer, SpeechConfig, SpeechSynthesisOutputFormat, AudioConfig
from moviepy.editor import AudioFileClip, concatenate_audioclips

def combine_audio_files(input_files, output_file):
    # Load all audio files into a list of AudioFileClip objects
    audio_clips = [AudioFileClip(file) for file in input_files]

    # Concatenate all audio clips into one
    combined_audio = concatenate_audioclips(audio_clips)

    # Write the result to the output file
    combined_audio.write_audiofile(output_file)

# Azure TTS subscription key and region
subscription_key = 'f74fe7fc6f6a48879e486a5b33e1653d'
region = 'westus'

# Configure the speech synthesis
speech_config = SpeechConfig(subscription=subscription_key, region=region)
speech_config.set_speech_synthesis_output_format(SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)

# Voice and supported styles dictionary
voices_and_styles = {
    'de-DE-ConradNeural': ['cheerful'],
    'en-GB-RyanNeural': ['chat', 'cheerful'],
    'en-GB-SoniaNeural': ['cheerful', 'sad'],
    'en-IN-NeerjaNeural': ['cheerful', 'empathetic', 'newscast'],
    'en-US-AriaNeural': ['angry', 'chat', 'cheerful', 'customerservice', 'empathetic', 'excited', 'friendly', 'hopeful',
                         'narration-professional', 'newscast-casual', 'newscast-formal', 'sad', 'shouting', 'terrified',
                         'unfriendly', 'whispering'],
    'en-US-DavisNeural': ['angry', 'chat', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
                          'unfriendly', 'whispering'],
    'en-US-GuyNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'newscast', 'sad', 'shouting',
                        'terrified', 'unfriendly', 'whispering'],
    'en-US-JaneNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
                         'unfriendly', 'whispering'],
    'en-US-JasonNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
                          'unfriendly', 'whispering'],
    'en-US-JennyNeural': ['angry', 'assistant', 'chat', 'cheerful', 'customerservice', 'excited', 'friendly', 'hopeful',
                          'newscast', 'sad', 'shouting', 'terrified', 'unfriendly', 'whispering'],
    'en-US-KaiNeural': ['conversation'],
    'en-US-LunaNeural': ['conversation'],
    'en-US-NancyNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
                          'unfriendly', 'whispering'],
    'en-US-SaraNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
                         'unfriendly', 'whispering'],
    'en-US-TonyNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
                         'unfriendly', 'whispering']
}


# Function to generate SSML for a given segment with specified style
def generate_ssml(start_time, end_time, text, voice, style):
    duration = end_time - start_time
    words_per_minute = 160  # Average speaking rate
    num_words = len(text.split())
    actual_duration = (num_words / words_per_minute) * 60  # in seconds
    speaking_rate = actual_duration / duration

    ssml = f'''
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
        <voice name="{voice}">
            <mstts:express-as style="{style}">
                <prosody rate="{speaking_rate:.2f}">{text}</prosody>
            </mstts:express-as>
        </voice>
    </speak>
    '''
    return ssml


# Function to synthesize speech from SSML
def synthesize_ssml_to_speech(ssml):
    synthesizer = SpeechSynthesizer(speech_config=speech_config)
    result = synthesizer.speak_ssml_async(ssml).get()
    audio_stream = AudioDataStream(result)
    return audio_stream


# Streamlit UI
# st.title('Text-to-Speech Generator')
#
# # Input field for captions
# captions_input = st.text_area('Enter the captions as JSON', height=200)
#
# # Dropdown menu for selecting voice
# selected_voice = st.selectbox('Select Voice', list(voices_and_styles.keys()))
#
# # Dropdown menu for selecting style based on selected voice
# selected_style = st.selectbox('Select Speaking Style', voices_and_styles[selected_voice])
#
# generate_button = st.button('Generate Speech')


# Function to generate TTS for all segments
def generate_tts_for_segments(captions, voice, style):
    all_voice_list = []
    for segment in captions:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']

        ssml = generate_ssml(start_time, end_time, text, voice, style)
        audio_stream = synthesize_ssml_to_speech(ssml)
        audio_file = f'{int(time.time())}_speech_{start_time}_{end_time}.wav'  # Unique filename based on timestamp
        audio_stream.save_to_wav_file(audio_file)
        all_voice_list.append(audio_file)
    audio_file = f'{int(time.time())}_speech_{start_time}_{end_time}.wav'
    combine_audio_files(all_voice_list, audio_file)
    return audio_file

client = AzureOpenAI(
            azure_endpoint="https://open-ai-east-us-2.openai.azure.com/",
            api_key="777a11c72ed74d45aa8d8abf92c87d19",
            api_version="2023-05-15")

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import os
from moviepy.editor import TextClip, CompositeVideoClip, ColorClip
import moviepy.config as mp_config
from voice_extraction import Conv_TDF_net_trimm, KimVocal
import torch
from scipy.io.wavfile import write

def remove_background_music(music_array, samplerate):

    music_tensor = torch.tensor(music_array, dtype=torch.float32)

    ONNX_MODEL_PATH = "Kim_Vocal.onnx"

    model_raw_python = Conv_TDF_net_trimm(
        model_path=ONNX_MODEL_PATH,
        use_onnx=True,
        target_name="vocals",
        L=11,
        l=3,
        g=48,
        bn=8,
        bias=False,
        dim_f=11,
        dim_t=8,
    )

    kimvocal = KimVocal()
    vocals_tensor = kimvocal.demix_vocals(
        music_tensor=music_tensor,
        sample_rate=samplerate,
        model=model_raw_python,
    )

    return vocals_tensor.numpy()



def add_caption_to_frame(frame, text, text_color, bg_color, is_bold, is_italic, horizontal_alignment):
    height, width = frame.shape[:2]

    # Choose font based on bold and italic settings
    if is_bold and is_italic:
        font = cv2.FONT_HERSHEY_TRIPLEX | cv2.FONT_ITALIC
    elif is_bold:
        font = cv2.FONT_HERSHEY_TRIPLEX
    elif is_italic:
        font = cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC
    else:
        font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = max(width, height) / 1500 #1
    thickness = max(1, int(font_scale * (2 if is_bold else 1))) #2 if is_bold else 1
    line_type = cv2.LINE_AA

    # Convert colors to tuples if they're not already
    text_color = tuple(map(int, text_color))
    bg_color = tuple(map(int, bg_color))

    # Split text into multiple lines
    max_width = int(width * 0.8)
    words = text.split()
    lines = []
    current_line = []
    current_width = 0

    for word in words:
        word_size = cv2.getTextSize(word + " ", font, font_scale, thickness)[0]
        if current_width + word_size[0] <= max_width:
            current_line.append(word)
            current_width += word_size[0]
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_width = word_size[0]

    if current_line:
        lines.append(" ".join(current_line))

    # Calculate text block dimensions
    text_height = sum([cv2.getTextSize(line, font, font_scale, thickness)[0][1] for line in lines])
    line_height = cv2.getTextSize("Tg", font, font_scale, thickness)[0][1]
    padding = int(max(10, font_scale * 20)) #20
    block_height = text_height + (len(lines) + 1) * padding
    block_width = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines]) + 2 * padding

    # Position the text block
    y_position = int(height * 0.7) - block_height // 2

    # Draw background rectangle covering the text region
    x_start = (width - block_width) // 2 if horizontal_alignment == "center" else (
        padding if horizontal_alignment == "left" else width - block_width - padding)
    cv2.rectangle(frame, (x_start, y_position), (x_start + block_width, y_position + block_height), bg_color, -1)

    # Draw text
    for line in lines:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        if horizontal_alignment == "left":
            x_position = x_start + padding
        elif horizontal_alignment == "right":
            x_position = width - text_size[0] - padding
        else:  # center
            x_position = (width - text_size[0]) // 2

        y_position += line_height + padding
        cv2.putText(frame, line, (x_position, y_position), font, font_scale, text_color, thickness, line_type)

    return frame


def process_frame(get_frame, t, text_color, bg_color, is_bold, is_italic, horizontal_alignment, captions):
    frame = get_frame(t).astype(np.uint8).copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
    current_time = t

    for caption in captions:
        if caption["start"] <= current_time < caption["end"]:
            frame = add_caption_to_frame(frame, caption["text"], text_color, bg_color, is_bold, is_italic,
                                         horizontal_alignment)
            break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert back to RGB
    return frame

def parse_captions(captions_string):
    # Regular expression to match the pattern of each caption entry
    pattern = re.compile(r'start:(\d+(\.\d+)?), end:(\d+(\.\d+)?), text: (.*?)(?=; start:|$)')
    matches = pattern.findall(captions_string)

    captions = []
    for match in matches:
        start = float(match[0])
        end = float(match[2])
        text = match[4].strip()
        captions.append({"start": start, "end": end, "text": text})

    return captions


def create_video_with_text(text, duration, output_path):
    os.environ["IMAGEMAGICK_BINARY"] = "/usr/bin/convert"

    # Set up the video properties
    width, height = 720, 1280  # Dimensions for the video
    background_color = (0, 0, 0)  # White background
    fontsize = 56  # Adjusted font size for the text
    txt_color = 'white'  # Text color

    # Create a background clip
    background = ColorClip(size=(width, height), color=background_color, duration=duration)

    # Create a text clip with the "Ultra-Regular" font
    txt_clip = TextClip(text, fontsize=fontsize, color=txt_color, font='DejaVu-Sans-Bold', size=(width, height),
                        method='caption', align='center').set_duration(duration)

    # Overlay the text clip on the background clip
    video = CompositeVideoClip([background, txt_clip])

    silent_audio = AudioFileClip("silent_3_seconds.wav")
    # Add the silent audio to the video
    video = video.set_audio(silent_audio)

    # Write the video to a file
    video.write_videofile(output_path, fps=24, codec='libx264')
    video.close()

# Example usage
# text = "Click the link below to get started"
# duration = 10  # Duration in seconds
# output_path = "output10_video.mp4"
# create_video_with_text(text, duration, output_path)

# mp_config.change_settings({"IMAGEMAGICK_BINARY": "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})

# def remove_audio_noise(video_path, output_path):
#     # Load video
#     video = VideoFileClip(video_path)
#     original_audio = video.audio
#     audio_file_path = "mywav.wav"
#     original_audio.write_audiofile(audio_file_path, codec='pcm_s16le')
#     rate, data = wavfile.read("mywav.wav")
#     print(rate)
#     orig_shape = data.shape
#     data = np.reshape(data, (2, -1))
#     # perform noise reduction
#     reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8, stationary=True, n_fft=512, hop_length=128,
#                                     win_length=512, use_tqdm=False)
#     wavfile.write("mywav_reduced_noise.wav", rate, reduced_noise.reshape(orig_shape))
#     # Set the new audio to the video
#     video = video.set_audio(AudioFileClip("mywav_reduced_noise.wav"))
#
#     # Write the output video file
#     video.write_videofile(output_path, codec='libx264', audio_codec='aac')


def overlay_captions(video_path, transcript, output_path):
    # Load the video
    video = VideoFileClip(video_path)

    # Parse the transcript
    captions = []
    for line in transcript.split(";"):
        parts = line.strip().split(", ")
        start = float(parts[0].split(":")[1])
        end = float(parts[1].split(":")[1])
        text = parts[2].split(": ")[1]

        # Create the text clip
        text_clip = TextClip(text, fontsize=24, color='black', bg_color='white', size=(video.w, None), method='caption')
        text_clip = text_clip.set_start(start).set_end(end).set_position(("center", "bottom"))

        captions.append(text_clip)

    # Create the final video with overlays
    final_video = CompositeVideoClip([video] + captions)

    # Write the result to a file
    final_video.write_videofile(output_path, codec='libx264')


# Example usage
# video_path = "input_video.mp4"
# transcript = "start:0, end:12.99, text: If you're tired of doing a lot of research into each game before you make your educated sports bet on it, let me tell you about an app called Odds R that can help you when making these decisions.; start:12.99, end:21.88, text: It was able to help me to figure out the best bets to make especially when I feel like going to look at the stats and see if this person will be playing better than this person, you know all that stuff.; start:21.88, end:30.75, text: The purpose of the app is to make sure that you get money and not lose it on just making wild guesses or wild bets.; start:30.75, end:39.62, text: This is an excellent app to help you win more and lose less. Odds R, check them out and stop wasting your time on all these devices trying to make sure that you are doing the best you can to make an educated guess on games."
# output_path = "output_video.mp4"
#
# overlay_captions(video_path, transcript, output_path)


def get_minimum_specs(video_paths):
    min_fps = float('inf')
    min_resolution = (float('inf'), float('inf'))

    for path in video_paths:
        clip = VideoFileClip(path)
        min_fps = min(min_fps, clip.fps)
        min_resolution = (min(min_resolution[0], clip.size[0]), min(min_resolution[1], clip.size[1]))
        clip.close()

    return min_fps, min_resolution

def create_caption(text, start, end, video_width, video_height):
    txt_clip = TextClip(text, fontsize=24, color='black', font='Arial', bg_color='white', align='center')
    txt_clip = txt_clip.set_position(('center', video_height * 0.75)).set_duration(end - start).set_start(start)
    return txt_clip


def same_original_file(segment1, segment2):
    def extract_original_name(segment):
        match = re.match(r"^(.*)-(\d+)\.mp4$", segment)
        if match:
            return match.group(1)
        else:
            return "Error"

    # Extract the original filenames from both segments
    original1 = extract_original_name(segment1)
    original2 = extract_original_name(segment2)

    # Compare the original filenames
    return original1 == original2

# def process_videos(video_paths, output_path, captions_flag, caption_data, background_music_array, ai_voiceover, selected_voice, selected_style, aspect_ratio, conversion_type, transition_duration=0):
#     import librosa
#     min_fps, min_resolution = get_minimum_specs(video_paths)
#     clips = []

#     for video_path in video_paths:
#         clip = VideoFileClip(video_path)

#         # clip = clip.resize(newsize=min_resolution)
#         if conversion_type == "Add Padding":
#             clip = convert_with_padding(clip, aspect_ratio)
#         elif conversion_type == "Blurred Background":
#             clip = convert_with_blur(clip, aspect_ratio)
#         else:
#             clip = convert_with_cropping(clip, aspect_ratio)
#         clip = clip.set_fps(min_fps)

#         # if transition_duration > 0:
#         #     clip = clip.set_end(clip.duration - 0)

#         clips.append(clip)

#     final_clip = concatenate_videoclips(clips, method="compose")  # , method="compose"

#     if not ai_voiceover:
#         audio_clips = [clip.audio for clip in clips]
#         concatenated_audio = concatenate_audioclips(audio_clips)
#         concatenated_audio.write_audiofile(output_path+"original_audio_before_bgremove.wav")
#         music_array, sample_rate = librosa.load(output_path+"original_audio_before_bgremove.wav", mono=False, sr=44100)
#         output_audio_array = remove_background_music(music_array, sample_rate)
#         print(output_audio_array.shape)
#         original_audio_array = output_audio_array.T
#     else:
#         generated_audio = generate_tts_for_segments(caption_data, selected_voice, selected_style)
#         music_array, sample_rate = librosa.load(generated_audio, mono=False,
#                                                 sr=44100)
#         original_audio_array = music_array.T

#     video_duration = final_clip.duration
#     background_music_duration = len(background_music_array) / 44100

#     if background_music_duration < video_duration:
#         repeats = int(video_duration // background_music_duration) + 1
#         background_music_array = np.tile(background_music_array, (repeats, 1))
#     background_music_array = background_music_array[:len(original_audio_array)]

#     # Reduce the volume of the background music (adjust as needed)
#     background_music_array = background_music_array * 0.15

#     # Combine the original audio and background music
#     combined_audio_array = original_audio_array + background_music_array.astype(np.float32)

#     max_val = np.max(np.abs(combined_audio_array))  # New line
#     print(max_val)
#     if max_val > 1.0:  # New line
#         combined_audio_array = combined_audio_array / max_val
#     # write(output_path+"original_audio_after_bgremove.wav", sample_rate, original_audio_array)
#     write(output_path+"combined_audio.wav", sample_rate, combined_audio_array)
#     audio_clip = AudioFileClip(output_path+"combined_audio.wav")

#     # Set the audio of the video clip
#     final_clip = final_clip.set_audio(audio_clip)

#     if captions_flag:
#         print("Adding captions")
#         print(type(final_clip))
#         final_clip_captions = final_clip.fl(lambda gf, t: process_frame(gf, t, (0, 0, 0), (255, 255, 255), True, False, "center", caption_data))
#         final_clip_captions.write_videofile(output_path, codec="libx264", audio_codec="aac")
#         final_clip_captions.close()
#         final_clip.close()
#     else:
#         final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
#         final_clip.close()

def replace_visuals_with_filler(main_video_path, filler_clips_data, output_path):
    """
    Replace visuals in a main video with filler clips while keeping the original audio.

    Parameters:
    - main_video_path (str): Path to the main video file.
    - filler_clips_data (list): List of tuples, each containing:
        - filler_clip_path (str): Path to a filler clip.
        - start_time (float): Start time in seconds in the main video where the filler should begin.
        - end_time (float): End time in seconds in the main video where the filler should end.
    - output_path (str): Path to save the output video.

    Example of filler_clips_data:
    [
        ("filler_clip1.mp4", 5, 10),  # filler clip from 5 to 10 seconds
        ("filler_clip2.mp4", 20, 25)  # filler clip from 20 to 25 seconds
    ]
    """
    # Load the main video
    main_video = VideoFileClip(main_video_path)

    # Create a list of segments (clips with or without filler)
    segments = []

    # Initialize the current time to start from the beginning of the main video
    current_time = 0

    for filler_clip_path, start_time, end_time in filler_clips_data:
        # Add the original part of the video before the filler (if any)
        if current_time < start_time:
            original_segment = main_video.subclip(current_time, start_time)
            segments.append(original_segment)

        # Load the filler clip
        filler_clip = VideoFileClip(filler_clip_path).subclip(0, end_time - start_time)

        # Resize the filler clip to match the main video's resolution
        filler_clip = filler_clip.resize(main_video.size)

        # Ensure filler clip duration matches the defined range
        filler_clip = filler_clip.set_duration(end_time - start_time)

        # Add the filler clip
        segments.append(filler_clip)

        # Update the current time to the end of the filler
        current_time = end_time

    # Add the remaining part of the main video after the last filler (if any)
    if current_time < main_video.duration:
        remaining_segment = main_video.subclip(current_time, main_video.duration)
        segments.append(remaining_segment)

    # Concatenate all segments (with fillers and original clips)
    final_video = concatenate_videoclips(segments)

    # Add the original audio from the main video
    final_video = final_video.set_audio(main_video.audio)

    # Write the output video file
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

    # Close video clips to free up resources
    final_video.close()
    # remaining_segment.close()
    # original_segment.close()
    main_video.close()
    filler_clip.close()

    for clip in segments:
        clip.close()

import ast
def parse_string_to_dict(input_string):
    # Removing curly braces at the start and end
    input_string = input_string.strip('{}')

    # Splitting the string by first comma to separate the two keys
    source_video_part, filler_clips_part = input_string.split(",filler_clips_data:")

    # Extracting source_video_name and removing unnecessary characters
    source_video_name = source_video_part.split(":")[1].strip('"')

    # Evaluating the list of tuples for filler_clips_data
    filler_clips_data = ast.literal_eval(filler_clips_part)

    # Constructing the dictionary
    result_dict = {
        'source_video_name': source_video_name,
        'filler_clips_data': filler_clips_data
    }
    
    return result_dict

def ad_one_influencer_hero(user_requirement_dict, final_output, details_text, output_path, process_videos):
    if process_videos:
        one_influencer_hero_prompt = """You are the best video editor out there in the industry. You will be given a detailed metadata about segments that come from different source videos. The naming of segments is in the format: "<source_video_name>-<scene_number>.mp4" Your role is to select the best source video according to the user requirement. Then, in that source video you need to suggest best visual edits using segment from other videos. You have to output what part of the video can be replaced by visuals from segments coming from other source videos.
        Output a python dictionary with the following two keys. The dict will be converted to JSON so format accordingly.
        - source_video_name (str): Path to the best matching source video file. Take in all segmentsof it into consideration. For eg. If tmpshcg6vsr-0.mp4, tmpshcg6vsr-1.mp4, tmpshcg6vsr-2.mp4 best match the user requirements, output tmpshcg6vsr.mp4
        - filler_clips_data (list): List of tuples, each containing:- filler_clip_path (str): Path to a filler clip. - start_time (float): Start time in seconds in the main video where the filler should begin. - end_time (float): End time in seconds in the main video where the filler should end. These segments should come from videos other than the source_video_name and decided using the matching visual aspects.
        For eg. 
            {source_video_name:"tmpshcg6vsr.mp4",
        filler_clips_data:[("tmp0hklcbic-0.mp4", 5, 8), ("tmp0hklcbic-1.mp4", 20, 27)]}
        Analyze all segments and output the source_video_name which is without the scene-number
        Strictly output only the dict and nothing else. Don't give any exaplanation in the output. Make sure there are no newlines in the list inside the dictionary as it will be parsed on JSON.
    """
        metadata_prompt = """Here is the list of user requirements: \n""" + dict_to_string(user_requirement_dict) +  """\nHere is the list of available segments with metadata: \n""" + details_text + """\n\nHere is the output that the Ad Agent had generated: \n""" + final_output
        response = client.chat.completions.create(
            model="gpt-4o-global",
            messages=[
                {"role": "system", "content": one_influencer_hero_prompt},
                {"role": "user", "content": metadata_prompt}
            ],
            temperature=0.5,
        )
        intermediate_data = response.choices[0].message.content

        print(response.usage)
        print(intermediate_data)
        ispython = True if intermediate_data[:9] == "```python" else False
        if ispython:
            intermediate_data = intermediate_data[10:-3]
        isjson = True if intermediate_data[:7] == "```json" else False
        if isjson:
            intermediate_data = intermediate_data[8:-3]
        print(intermediate_data)
        final_output = ast.literal_eval(intermediate_data)

        replace_visuals_with_filler("/tmp/"+os.path.splitext(final_output["source_video_name"])[0], final_output["filler_clips_data"], output_path)
    return True, {"updated_files":[], "updated_script":"", "summary":"", "final_quality_score":0.85}

from moviepy.audio.fx.all import audio_normalize
from moviepy.editor import vfx, ImageClip, AudioClip
import traceback

# from moviepy.editor import  freeze
def process_videos(video_paths, output_path, captions_flag, caption_data, background_music_array, ai_voiceover, selected_voice, selected_style, aspect_ratio, conversion_type, transition_duration=0):
    import librosa
    min_fps, min_resolution = get_minimum_specs(video_paths)
    clips = []
    converters = []

    try:
        for video_path in video_paths:
            clip = VideoFileClip(video_path)

            # Initialize the converter with the clip and the target aspect ratio
            converter = VideoConverterWithBlur(clip, aspect_ratio)
            converters.append(converter)

            # Convert the clip using the converter instance
            clip = converter.convert()
            freeze_duration = 5
            # Set the FPS of the clip
            clip = clip.set_fps(min_fps)
            clips.append(clip)

    # Extend the last frame of the current clip by 0.2 seconds
    #         if len(video_paths) - video_paths.index(video_path) <= 2:
    #             freeze_frame = clip.get_frame(clip.duration - 0.1)  # Get the last frame
    #             print("This is the frozen frame", freeze_frame)
    #     # Create an ImageClip of the last frame with a 0.2-second duration
    #             frozen_clip = ImageClip(freeze_frame).set_duration(0.29).set_fps(clip.fps)
    #             silent_audio = AudioClip(lambda t: [0], duration=0.1)

    # # Add the silent audio to the frozen clip
    #             frozen_clip = frozen_clip.set_audio(silent_audio)
    #             print("Frozen Clip created")
    #     # Append the frozen clip to the list
    #             clips.append(frozen_clip)
            # Append the converted clip to the clips list
            # clips.append(clip)
            # extended_clip = clip.fx(vfx.freeze, t='end', duration=5)
            # clips.append(extended_clip)

        # Concatenate the video clips
        final_clip = concatenate_videoclips(clips, method="compose")

        if not ai_voiceover:
            audio_clips = [clip.audio for clip in clips]
            concatenated_audio = concatenate_audioclips(audio_clips)
            # concatenated_audio = audio_normalize(concatenated_audio_temp)

        # Set the audio of the video clip
        # final_clip = final_clip.set_audio(normalized_audio)
            concatenated_audio.write_audiofile(output_path+"original_audio_before_bgremove.wav")
            music_array, sample_rate = librosa.load(output_path+"original_audio_before_bgremove.wav", mono=False, sr=44100)
            output_audio_array = remove_background_music(music_array, sample_rate)
            print(output_audio_array.shape)
            original_audio_array = output_audio_array.T
        else:
            generated_audio = generate_tts_for_segments(caption_data, selected_voice, selected_style)
            music_array, sample_rate = librosa.load(generated_audio, mono=False,
                                                    sr=44100)
            original_audio_array = music_array.T

        video_duration = final_clip.duration
        background_music_duration = len(background_music_array) / 44100

        if background_music_duration < video_duration:
            repeats = int(video_duration // background_music_duration) + 1
            background_music_array = np.tile(background_music_array, (repeats, 1))
        background_music_array = background_music_array[:len(original_audio_array)]

        # Reduce the volume of the background music (adjust as needed)
        background_music_array = background_music_array * 0.11

        # Combine the original audio and background music
        combined_audio_array = original_audio_array + background_music_array.astype(np.float32)

        max_val = np.max(np.abs(combined_audio_array))
        print(max_val)
        if max_val > 1.0:
            combined_audio_array = combined_audio_array / max_val

        write(output_path+"combined_audio.wav", sample_rate, combined_audio_array)
        audio_clip = AudioFileClip(output_path+"combined_audio.wav")
        normalized_audio = audio_normalize(audio_clip)

        # Set the audio of the video clip
        final_clip = final_clip.set_audio(normalized_audio)

        # Handle captions if needed
        if captions_flag:
            print("Adding captions")
            print(type(final_clip))
            final_clip_captions = final_clip.fl(lambda gf, t: process_frame(gf, t, (0, 0, 0), (255, 255, 255), True, False, "center", caption_data))
            final_clip_captions.write_videofile(output_path, codec="libx264", audio_codec="aac")
            final_clip_captions.close()
        else:
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        # Close the final clip and all converter objects to ensure resources are released
        if 'final_clip' in locals():
            final_clip.close()
        
        for converter in converters:
            converter.close()


def generate_details_text(df):
    details = []
    for index, row in df.iterrows():
        row_details = (
            f"Filename: {row['filenames']}\n"
            f"Category: {row['categories']}\n"
            f"Duration: {row['segment_durations']}\n"
            f"Transcript: {row['transcript']}\n"
            f"Scene Description: {row['sceneDescription']}\n"
            # f"Age: {row['age']}\n"
            f"Gender: {row['gender']}\n"
            # f"Shot Location: {row['shotLocation']}\n"
            f"Aesthetic Score: {row['aestheticScore']}\n"
            f"Is there a Caption Overlay?: {row['isCaption']}\n"
            # f"Audio Tone: {row['audio_tone']}\n"
        )
        details.append(row_details)
    return "\n".join(details)

# def generate_details_text(df):
#     details = []
#     for index, row in df.iterrows():
#         row_details = (
#             f"Filename: {row['filenames']}\n"
#             f"Duration: {row['segment_durations']}\n"
#             f"Transcript: {row['transcript']}\n"
#             f"Scene Description: {row['sceneDescription']}\n"
#             f"Gender: {row['gender']}\n"
#             f"Aesthetic Score: {row['aestheticScore']}\n"
#             f"Is there a Caption Overlay?: {row['isCaption']}\n"
#         )
        
#         # Only add the Category if it equals 'cta'
#         if row['categories'] == 'cta':
#             row_details += f"Category: {row['categories']}\n"
        
#         details.append(row_details)
    
#     return "\n".join(details)


# Generate and store the details text in a variable
# details_text = generate_details_text(df)

# Now details_text contains the formatted text for all rows
# print(details_text)
def dict_to_string(d):
    return '\n'.join(f'{key}: {value}' for key, value in d.items())

# One Influencer Hero: Ideal for a single, powerful influencer whose message aligns with the ad's objective. Strictly, it should have only one actor and all the segments should come from one source video. Segments without a person can come from any source video. Otherwise, don't output it as a category.
def get_themes(df):
    import ast
    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are an expert Ad theme creator. You will get metadata about different Segments. You have to output themes of new Ads that can be generated by stitiching these segments. The themes should be unique and best-suited for performance marketing. These themes should highlight product/lifestyle, scenes, important objects. Each Ad is made up of atleast 4-5 segments. Based on the input segment metadata, show how many segments from the given data relate to a particular concept. Just output all the possible concepts in a python list format. Make sure to use double quotes in each element of the dict inside list and that elements are separated by comma. Make sure the themes are not redundant. Strictly don't output any description or segment filenames related to the themes.
                 These are the possible values of narration style:
                Testimonial Style: Best when multiple influencers provide strong social proof through positive experiences.
                Lead with One, Transition to Others: Start with a compelling influencer and build a multi-faceted narrative with additional voices.
                 For eg.
                 
                 [{"Concept 1": "Embrace the Energy",
"Core Idea": "A dynamic montage showcasing outdoor activities and the energizing effects of our protein bar.",
"Key Visuals": "Running, yoga, nature, protein bar consumption, bright colors, energetic camera movements",
"Recommended Music": "Upbeat music, nature sounds",
"Recommended Emotional Tone": "voiceover emphasizing natural energy, Uplifting, empowering",
"CTA": "Fuel your active life.",
"Actors": "Female",
"No. of segments related to this concept": "7",
"Narration Style": "Testimonial Style"},
                 
{"Concept 2": "Wholesome Fuel for Every Adventure"
"Core Idea": "Highlight the protein bar as a healthy snack for outdoor adventures.",
"Key Visuals": "Hiking, biking, picnics, natural settings, focus on the protein bar's portability",
"Recommended Music": "Acoustic guitar music, nature sounds",
"Recommended Emotional Tone": "friendly voiceover, Adventurous, lighthearted",
"CTA": "Grab a bar and go!",
"Actors": "Male, Female",
"No. of segments related to this concept": "10"
"Narration Style": "Lead with One, Transition to Others"}]
"""},
                {"role": "user", "content": "Here the metadata of segments: " +  generate_details_text(df)}
            ],
            temperature=0.5,
        )
    intermediate_data = response.choices[0].message.content
    print(intermediate_data)
    ispython = True if intermediate_data[:9]=="```python" else False
    if ispython:
        intermediate_data = intermediate_data[10:-3]
    list_of_dicts = ast.literal_eval(intermediate_data)
    list_of_strings = [str(d) for d in list_of_dicts]
    return list_of_strings

def no_of_related_segments(df, list_of_concepts):
    import ast
    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You will get metadata about different Segments. You have to output the no. of segments that are related with each concept. Output concept title, no. of related segments and related segment filenames and scene description and transcript for the related segments.
"""},
                {"role": "user", "content": "Here are the concepts for which we need to find the no. of related segments: " + "\n".join(list_of_concepts) + "Here the metadata of segments: " +  generate_details_text(df)}
            ],
            temperature=0.5,
        )
    intermediate_data = response.choices[0].message.content
    print(intermediate_data)
    ispython = True if intermediate_data[:9]=="```python" else False
    if ispython:
        intermediate_data = intermediate_data[10:-3]
    
    return intermediate_data

def create_ad(df, user_requirement_dict):
    transactional_prompt = """You are a pro video editor and an expert in making ads. You will be given bite sized segments with metadata and you will have to select the best segments that can be merged together to create perfect advertisements. 
     You know the flow of events for Ads that are as follows:
     Viral Ad: Hook, Problem (Buildup), Solution, Social Proof, Call to Action.
     Strictly adhere to the User requirements. The output should be a list of segment filenames that when concatenated will give the best cohesive output ad around 15s. IMPORTANT: The most important thing is that the final Ad should look complete and cohesive. It should flow and look like one complete Ad. It should not have abrupt ending. It is okay if you break the order of segment categories or the Final Ad is longer. You can output any no. files if needed to fill any gaps. Also, write the script (that will be converted to audio) for this Ad. You will output the filenames and transcript in a python dict format. Always Include segments from multiple files.
     Example Output:
     {
        "files": ["file1.mp4", "file8.mp4", "file3.mp4"],
        "Script": "This is a sample script",
     }
     Just output the dict and nothing else.
    """
    details_text = generate_details_text(df)
    metadata_prompt = """Here is the list of user requirements: \n""" + dict_to_string(user_requirement_dict) + """\nHere is the list of available segments with metadata: \n""" + details_text
    # print(transactional_prompt)
    # print(metadata_prompt)
    response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": transactional_prompt},
                    {"role": "user", "content": metadata_prompt}
                ],
                temperature=0.5,
            )
    intermediate_data = response.choices[0].message.content

    print(response.usage)
    ispython = True if intermediate_data[:9]=="```python" else False
    if ispython:
        intermediate_data = intermediate_data[10:-3]
    print(intermediate_data)
    final_output = intermediate_data #json.loads(intermediate_data)
    # list_of_videos = eval(intermediate_data)
    # process_videos(final_output["files"], "FinalTransactionalAd.mp4")
    return user_requirement_dict ,final_output, details_text



# def ad_critic(user_requirement_dict, final_output, details_text, filename, cta_text, is_process_videos, iscaption, is_ai_voiceover, selected_voice, selected_style, aspect_ratio, conversion_type, temperature=0.5):
#     critic_prompt = """You are the best Ad critic out there in the business. The Ads that you finalize are used for Meta Ad Library for campaigns of Fortune 500 companies. So you make sure that it is of highest quality. I have an agent that works on creating Ads by merging bite sized segments. \
#                     I will provide the metadata it had, and the files it selected for making the best Ad. The loose structure of an Ad is: Hook, Problem (Buildup), Solution, Social Proof, Call to Action.
#                     You have to tell whether the given Ad is good to be given as the final output to the Fortune 500 company. If not, suggest changes or mention that an Ad is simply not possible out of the given segments. Make sure and take utmost care that the final Ad is cohesive and and the duration is according to the user's requirement.
#                     It is important to note that these segments come from different original files. The naming convention of segments is {original_filename}-{segment_number}.mp4. STRICTLY - Each Ad should be a combination of segments from multiple original files. If it's not possible to have segments from multiple files, write your own script.
#                     The output should be in a python dict format.
#                     Here is the definition of the output fields:
#                     "original_quality_score": A score between 0.0 to 1.0. The quality of the original Ad. 1.0 being the best.
#                     "critic" - Output "good" if the existing combination is perfect. Output "bad" if changes are required in the existing combination of segments. Output "none" if no good Ad is possible from the given segment content.
#                     "feedback" - This is the feedback on the final Ad. Mention the changes that you will be making and the reasons for the same. Also mention how you will be using segments from different original files.
#                     "updated_files" - Updated files list based on the above conditions.
#                     "updated_files_duration" - Duration of the updated files in seconds.
#                     "updated_script" - If the critic is "good", the updated_script should resemble the exact script from the existing segments. 
#                     If the critic is "bad" there can be two cases: 
#                     Case 1: Another combination of existing segments (using existing transcript) can be used to generate the best Ad. In this case, give the updated script (It should be a combination of transcripts from the original segments only. STRICTLY - DO NOT GENERATE AYTHING ON YOUR OWN). Also give the updated files that when merged together will create the best Ad. In this case, keep "is_script_from_segments" Yes.
#                     Case 2: No combination of existing transcripts make the best Ad OR The user hasn't liked the only possible combination OR All segments are coming from a single file. In this case write your own script (that will be converted to audio overlay), and update the files to select the best files that visually match the script (Use files information). IMPORTANT: Make sure to write script according to length of each segment. STRICTLY, the script should be around 2-2.5 words per second. Ensure same duration of the converted audio length to the final Ad duration from the files. In this case, keep "is_script_from_segments" No.
#                     If the critic is "none", keep the updated script empty.
#                     "is_script_from_segments" - "Yes" or "No" based on the above conditions. Make sure it is in double quotes
#                     "summary": This is the summary of the final Ad by linking it with user's input.
#                     "keywords": Keywords that we can get from the final Ad (both audio and visual).
#                     "features": A list of audio and visual features. for eg ["Tone: Excited", "Female Speaking", "Location: Outdoors"]
#                     "final_quality_score": A score between 0.0 to 1.0. This is the score of the quality of the Ad after the changes.
#                      For eg.
#                     {
#                     "original_quality_score": 0.45,
#                     "critic": "good",
#                     "feedback": "This is the feedback", 
#                     "updated_files": ["file1.mp4", "file5.mp4", file2.mp4""] // If review of existing is good, output the existing filenames. If bad output the updated filenames, Make sure the files go with the entire script i.e. there should be files for each sentence of the script. If none output an empty list.
#                     "updated_files_duration": [3.53, 6.5, 2.0], //Duration of the updated files in seconds
#                     "updated_script": "start:0, end:5, text: This is the updated script line 1; start:5, end:10, text: This is the updated script line 2;", //The start and end time are the time of when the segment starts and ends in the Final Ad. The start time of any segment is where the last segment ends. The end time of any segment will be (last segment end + its duration) and should be coherent with the visual clips
#                     "is_script_from_segments": "Yes" //Yes if the entire script is from the existing segments, No if you've written even one sentence of script on your own. Write your own script only if really required.
#                     "summary": "This is the final Ad summary"
#                     "keywords": ["keyword1", "keyword2", "keyword3"],
#                     "features": ["feature1", "feature2", "feature3"],
#                     "final_quality_score": 0.75,
#                     }
#                     Output only the dict and nothing else. Don't put comma after the last element in the dict. The dict will be parsed as JSON so make sure all charcters are escaped correctly if required.
#                     Things to keep in mind:
#                     1. Audio Visual Synergy
#                     2. Maximize the final_quality_score (Ad Impact).
#                     3. The final Ad's duration should STRICTLY be according to the user's requirement. If it is around 15 seconds use 3 segments. If it is around 20 seconds use 4 segments. If it is around 30 seconds, use at least 6 segments. If it is around 45 seconds, use at least 9 segments. If it is around 60 seconds, use at least 13 segments. If it is around 90 seconds, use at least 20 segments. If it is around 120 seconds, use at least 25 segments to fulfil the requirement. Use segments without transcript to fill in if required.
#                     4. According to user's requirement, focus more on just the product or overall lifestyle that involves that product.
#                     5. Cohesiveness- The pieces should make sense one after other as a whole. It should flow and look like one complete Ad from start to end. Review the script and segments to ensure there is no logical cutoff. IMPORTANT: Make sure it has a good start and good closing. Always Include segments from multiple files.
#                     6. Strictly adhere to all the user requirements for creating the final Ad. If that's not possible, don't create an Ad.
#                     7. If the user requests a One Influencer Hero narrative style, Strictly, it should have only one actor and all the segments should come from one source video. Segments without a person can come from any source video.
#                     8. Don't use segments without any transcript for Ad Creation that are longer than 5 seconds.
#                     9. The advertisement should srictly end with a Call to Action. A Call to Action (CTA) encourages an audience to take a specific action, like purchasing, subscribing, or engaging. Generally, they would appear at the end of the original videos.
#                     """
#     metadata_prompt = """Here is the list of user requirements: \n""" + dict_to_string(user_requirement_dict) +  """\nHere is the list of available segments with metadata: \n""" + details_text + """\n\nHere is the output that the Ad Agent had generated: \n""" + final_output
#     response = client.chat.completions.create(
#         model="gpt-4o-global",
#         messages=[
#             {"role": "system", "content": critic_prompt},
#             {"role": "user", "content": metadata_prompt}
#         ],
#         temperature=temperature,
#     )
#     intermediate_data = response.choices[0].message.content

#     print(response.usage)
#     # print(intermediate_data)
#     ispython = True if intermediate_data[:9] == "```python" else False
#     if ispython:
#         intermediate_data = intermediate_data[10:-3]
#     isjson = True if intermediate_data[:7] == "```json" else False
#     if isjson:
#         intermediate_data = intermediate_data[8:-3]
#     print(intermediate_data)
#     final_output = json.loads(intermediate_data)
    
#     if is_process_videos and final_output["is_script_from_segments"]=="No":
#         tts_prompt = """You will be given a string with segments of texts and the no. of words that each text should be converted to. Rewrite each text to meaningful quality sentences according to the no. of words given with it.
#         Here is the string:\n""" + final_output["updated_script"] + """\nHere is the word count per each segment: [""" + ', '.join(map(str, [round(x * 2.2) for x in final_output["updated_files_duration"]])) + """]\nOutput just the rewrited string in the exact same format with the changed text. Ensure that there are no newlines in the string and colon even at the end of last text."""
        
        
#         print(tts_prompt)

#         response = client.chat.completions.create(
#             model="gpt-4o-global",
#             messages=[
#                 {"role": "system", "content": tts_prompt},
#             ],
#             temperature=0.9,
#         )
#         intermediate_data = response.choices[0].message.content

#         print(intermediate_data)
#         final_output["updated_script"] = intermediate_data

#     try:
#         if final_output["updated_files"]:
#             if is_process_videos:
#                 # filename = "FinalTransactionalAd" + str(int(temperature*10)) +".mp4"
#                 create_video_with_text(cta_text,3,"cta.mp4")
#                 # print(final_output["updated_files"])
#                 # print(final_output["updated_files"].append("cta.mp4"))
#                 final_output["updated_files"].append("cta.mp4")
#                 print(final_output["updated_files"])
#                 background_music_clip = AudioFileClip("bgmusic.mp3")
#                 bgarray = background_music_clip.to_soundarray(fps=44100)
#                 process_videos(final_output["updated_files"], filename, iscaption, parse_captions(final_output["updated_script"]), bgarray, is_ai_voiceover, selected_voice, selected_style, aspect_ratio, conversion_type)

#             # overlay_captions("FinalTransactionalAd.mp4", final_output["updated_script"], "FinalTransactionalAdCaption.mp4")
#             return True, final_output
#         else:
#             return False, False
#     except Exception as e:
#         print(e)
#         return False, False




def ad_critic(user_requirement_dict, final_output, details_text, filename, cta_text, is_process_videos, iscaption, is_ai_voiceover, selected_voice, selected_style, aspect_ratio, conversion_type, m_value, temperature=0.5):
    critic_prompt = """You are the best Ad critic out there in the business. The Ads that you finalize are used for Meta Ad Library for campaigns of Fortune 500 companies. So you make sure that it is of highest quality. I have an agent that works on creating Ads by merging bite sized segments. \
                    I will provide the metadata it had, and the files it selected for making the best Ad. The loose structure of an Ad is: Hook, Problem (Buildup), Solution, Social Proof.
                    You have to tell whether the given Ad is good to be given as the final output to the Fortune 500 company. If not, suggest changes or mention that an Ad is simply not possible out of the given segments. Make sure and take utmost care that the final Ad is cohesive and and the duration is according to the user's requirement.
                    It is important to note that these segments come from different original files. The naming convention of segments is {original_filename}-{segment_number}.mp4. STRICTLY - Each Ad should be a combination of segments from multiple original files. If it's not possible to have segments from multiple files, write your own script.
                    The output should be in a python dict format.
                    Here is the definition of the output fields:
                    "original_quality_score": A score between 0.0 to 1.0. The quality of the original Ad. 1.0 being the best.
                    "critic" - Output "good" if the existing combination is perfect. Output "bad" if changes are required in the existing combination of segments. Output "none" if no good Ad is possible from the given segment content.
                    "feedback": This is the feedback on the final Ad. Mention the changes that you will be making and the reasons for the same. Also mention how you will be using segments from different original files.
                    "updated_files": Updated files list based on the above conditions.
                    "updated_files_duration": Duration of the updated files in seconds.
                    "updated_script": If the critic is "good", the updated_script should resemble the exact script from the existing segments. 
                    If the critic is "bad" there can be two cases: 
                    Case 1: Another combination of existing segments (using existing transcript) can be used to generate the best Ad. In this case, give the updated script (It should be a combination of transcripts from the original segments only. STRICTLY - DO NOT GENERATE AYTHING ON YOUR OWN). Also give the updated files that when merged together will create the best Ad. In this case, keep "is_script_from_segments" Yes.
                    Case 2: No combination of existing transcripts make the best Ad OR The user hasn't liked the only possible combination OR All segments are coming from a single file. In this case write your own script (that will be converted to audio overlay), and update the files to select the best files that visually match the script (Use files information). IMPORTANT: Make sure to write script according to length of each segment. STRICTLY, the script should be around 2-2.5 words per second. Ensure same duration of the converted audio length to the final Ad duration from the files. In this case, keep "is_script_from_segments" No.
                    If the critic is "none", keep the updated script empty.
                    "is_script_from_segments": "Yes" or "No" based on the above conditions. Make sure it is in double quotes
                    "summary": This is the summary of the final Ad by linking it with user's input.
                    "keywords": Keywords that we can get from the final Ad (both audio and visual).
                    "features": A list of audio and visual features. for eg ["Tone: Excited", "Female Speaking", "Location: Outdoors"]
                    "final_quality_score": A score between 0.0 to 1.0. This is the score of the quality of the Ad after the changes.
                     For eg.
                    {
                    "original_quality_score": 0.45,
                    "critic": "good",
                    "feedback": "This is the feedback", 
                    "updated_files": ["file1.mp4", "file5.mp4", file2.mp4""] // If review of existing is good, output the existing filenames. If bad output the updated filenames, Make sure the files go with the entire script i.e. there should be files for each sentence of the script. If none output an empty list.
                    "updated_files_duration": [3.53, 6.5, 2.0], //Duration of the updated files in seconds
                    "updated_script": "start:0, end:5, text: This is the updated script line 1; start:5, end:10, text: This is the updated script line 2;", //The start and end time are the time of when the segment starts and ends in the Final Ad. The start time of any segment is where the last segment ends. The end time of any segment will be (last segment end + its duration) and should be coherent with the visual clips
                    "is_script_from_segments": "Yes" //Yes if the entire script is from the existing segments, No if you've written even one sentence of script on your own. Write your own script only if really required.
                    "summary": "This is the final Ad summary"
                    "keywords": ["keyword1", "keyword2", "keyword3"],
                    "features": ["feature1", "feature2", "feature3"],
                    "final_quality_score": 0.75,
                    }
                    Output only the dict and nothing else. Don't put comma after the last element in the dict. The dict will be parsed as JSON so make sure all charcters are escaped correctly if required.
                    Things to keep in mind:
                    1. Audio Visual Synergy
                    2. Maximize the final_quality_score (Ad Impact).
                    3. The final Ad's duration should STRICTLY be according to the user's requirement. If it is around 15 seconds use 3 segments. If it is around 20 seconds use 4 segments. If it is around 30 seconds, use at least 6 segments. If it is around 45 seconds, use at least 9 segments. If it is around 60 seconds, use at least 13 segments. If it is around 90 seconds, use at least 20 segments. If it is around 120 seconds, use at least 25 segments to fulfil the requirement. Use segments without transcript to fill in if required.
                    4. According to user's requirement, focus more on just the product or overall lifestyle that involves that product.
                    5. Cohesiveness- The pieces should make sense one after other as a whole. It should flow and look like one complete Ad from start to end. Review the script and segments to ensure there is no logical cutoff. IMPORTANT: Make sure it has a good start and good closing. Always Include segments from multiple files.
                    6. Strictly adhere to all the user requirements for creating the final Ad. If that's not possible, don't create an Ad.
                    7. If the user requests a One Influencer Hero narrative style, Strictly, it should have only one actor and all the segments should come from one source video. Segments without a person can come from any source video.
                    8. Don't use segments without any transcript for Ad Creation that are longer than 5 seconds."""
    metadata_prompt = """Here is the list of user requirements: \n""" + dict_to_string(user_requirement_dict) +  """\nHere is the list of available segments with metadata: \n""" + details_text + """\n\nHere is the output that the Ad Agent had generated: \n""" + final_output
    response = client.chat.completions.create(
        model="gpt-4o-global",
        messages=[
            {"role": "system", "content": critic_prompt},
            {"role": "user", "content": metadata_prompt}
        ],
        temperature=temperature,
        # max_tokens=int(4000 * m_value)  # Adjust max_tokens based on m_value
    )
    intermediate_data = response.choices[0].message.content

    print(response.usage)
    # print(intermediate_data)
    ispython = True if intermediate_data[:9] == "```python" else False
    if ispython:
        intermediate_data = intermediate_data[10:-3]
    isjson = True if intermediate_data[:7] == "```json" else False
    if isjson:
        intermediate_data = intermediate_data[8:-3]
    print(intermediate_data)
    final_output = json.loads(intermediate_data)
    
    if is_process_videos and final_output["is_script_from_segments"]=="No":
        tts_prompt = """You will be given a string with segments of texts and the no. of words that each text should be converted to. Rewrite each text to meaningful quality sentences according to the no. of words given with it.
        Here is the string:\n""" + final_output["updated_script"] + """\nHere is the word count per each segment: [""" + ', '.join(map(str, [round(x * m_value) for x in final_output["updated_files_duration"]])) + """]\nOutput just the rewrited string in the exact same format with the changed text. Ensure that there are no newlines in the string and colon even at the end of last text."""
        
        
        print(tts_prompt)

        response = client.chat.completions.create(
            model="gpt-4o-global",
            messages=[
                {"role": "system", "content": tts_prompt},
            ],
            temperature=0.5,
        )
        intermediate_data = response.choices[0].message.content

        print(intermediate_data)
        final_output["updated_script"] = intermediate_data

    try:
        if final_output["updated_files"]:
            if is_process_videos:
                # filename = "FinalTransactionalAd" + str(int(temperature*10)) +".mp4"
                create_video_with_text(cta_text,3,"cta.mp4")
                # print(final_output["updated_files"])
                # print(final_output["updated_files"].append("cta.mp4"))
                final_output["updated_files"].append("cta.mp4")
                print(final_output["updated_files"])
                background_music_clip = AudioFileClip("bgmusic.mp3")
                bgarray = background_music_clip.to_soundarray(fps=44100)
                process_videos(final_output["updated_files"], filename, iscaption, parse_captions(final_output["updated_script"]), bgarray, is_ai_voiceover, selected_voice, selected_style, aspect_ratio, conversion_type)

            # overlay_captions("FinalTransactionalAd.mp4", final_output["updated_script"], "FinalTransactionalAdCaption.mp4")
            return True, final_output
        else:
            return False, False
    except Exception as e:
        print(e)
        return False, False


###########################################################
voiceover_flag_status = False
def status_voiceover_flag(voiceover_flag=False):
    return voiceover_flag

voiceover_flag_status = status_voiceover_flag()
def eliminate_talking_head(df,voiceover_flag_status):
    if voiceover_flag_status:
        talking_head_rows = df[df['talkingHead'] == True]
        return talking_head_rows
    else:
        return df

###########################################################

if __name__ == "__main__":
    df = pd.read_csv("segment_metadata2.csv")
    df = eliminate_talking_head(df,voiceover_flag_status)
    requirements, output, details_text = create_ad(df, {"Ad Description": "An advertisemnt Build a bear"})
    # requirements, output, details_text = create_ad(df, {"Selling Points of the product": "It gives the best odds fro betting", "Pain points of the customers": "It addresses the pain points of not getting the best odds and losing bets.", "Ad Description": "None", "Ad tone": "None", "Actor Gender Preference": "Any", "Shot Location Preference": "None"})
    # requirements, output, details_text = create_ad(df, {"Selling Points of the product": """FlavCity provides easy-to-follow, healthy recipes and meal prep ideas""", "Pain points of the customers": """Some customers find the recipes to be time-consuming and requiring specific, sometimes hard-to-find ingredients, which can be inconvenient.""", "Ad Description": "I want to create a video that showcases all the different types of food items possible with their protein powder. Strictly focus on the variety of items. Keep it quick and don't show the entire recipe.", "Ad tone": "Any", "Actor Gender Preference": "Any", "Shot Location Preference": "Any"})
    # ad_critic(requirements, output, details_text, 0.2)
    q,w = ad_critic(requirements, output, details_text, "firstad.mp4","This is CTA",True, True, 0.5)
    # a,b,c,d = ad_critic({"Ad Description": "An advertisemnt Build a bear", "Other Requirements": "An Ad was already made as a combination of the following files. Do not make a similar Ad. Make a different version now. Current Ad: " + ', '.join(r)}, output, details_text, 0.5)
    a, b = ad_critic({"Ad Description": "An advertisement on Build a bear", "Other Requirements": "The user did not like the ad that was already created. Create an entirely new and imapctful version while making sure to use segments from multiple videos."}, "\nFilenames: " + ', '.join(w["updated_files"]) + "\n Script: " + w["updated_script"], details_text, "secondad.mp4","This is CTA",True, True,  0.5)
    x, y = ad_critic({"Ad Description": "An advertisement on Build a bear", "Other Requirements": "The user did not like the ad that was already created. Create an entirely new and imapctful version while making sure to use segments from multiple videos."}, "\nFilenames: " + ', '.join(b["updated_files"]) + "\n Script: " + b["updated_script"], details_text, "thirdad.mp4","This is CTA",True, True, 0.5)

    # ad_critic(requirements, output, details_text, 0.5)
    # ad_critic(requirements, output, details_text, 0.9)



# from openai import AzureOpenAI
# import pandas as pd
# from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, TextClip, CompositeVideoClip, concatenate_audioclips
# import json
# import re
# import cv2
# import numpy as np
# import moviepy.config as mp_config
# from video_utils import convert_with_cropping, convert_with_padding, convert_with_blur, VideoConverterWithBlur

# # from tts import generate_tts_for_segments

# # import noisereduce as nr
# # from scipy.io import wavfile
# # import numpy as np

# import time
# import json
# from azure.cognitiveservices.speech import AudioDataStream, SpeechSynthesizer, SpeechConfig, SpeechSynthesisOutputFormat, AudioConfig
# from moviepy.editor import AudioFileClip, concatenate_audioclips

# def combine_audio_files(input_files, output_file):
#     # Load all audio files into a list of AudioFileClip objects
#     audio_clips = [AudioFileClip(file) for file in input_files]

#     # Concatenate all audio clips into one
#     combined_audio = concatenate_audioclips(audio_clips)

#     # Write the result to the output file
#     combined_audio.write_audiofile(output_file)

# # Azure TTS subscription key and region
# subscription_key = 'f74fe7fc6f6a48879e486a5b33e1653d'
# region = 'westus'

# # Configure the speech synthesis
# speech_config = SpeechConfig(subscription=subscription_key, region=region)
# speech_config.set_speech_synthesis_output_format(SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)

# # Voice and supported styles dictionary
# voices_and_styles = {
#     'de-DE-ConradNeural': ['cheerful'],
#     'en-GB-RyanNeural': ['chat', 'cheerful'],
#     'en-GB-SoniaNeural': ['cheerful', 'sad'],
#     'en-IN-NeerjaNeural': ['cheerful', 'empathetic', 'newscast'],
#     'en-US-AriaNeural': ['angry', 'chat', 'cheerful', 'customerservice', 'empathetic', 'excited', 'friendly', 'hopeful',
#                          'narration-professional', 'newscast-casual', 'newscast-formal', 'sad', 'shouting', 'terrified',
#                          'unfriendly', 'whispering'],
#     'en-US-DavisNeural': ['angry', 'chat', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
#                           'unfriendly', 'whispering'],
#     'en-US-GuyNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'newscast', 'sad', 'shouting',
#                         'terrified', 'unfriendly', 'whispering'],
#     'en-US-JaneNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
#                          'unfriendly', 'whispering'],
#     'en-US-JasonNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
#                           'unfriendly', 'whispering'],
#     'en-US-JennyNeural': ['angry', 'assistant', 'chat', 'cheerful', 'customerservice', 'excited', 'friendly', 'hopeful',
#                           'newscast', 'sad', 'shouting', 'terrified', 'unfriendly', 'whispering'],
#     'en-US-KaiNeural': ['conversation'],
#     'en-US-LunaNeural': ['conversation'],
#     'en-US-NancyNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
#                           'unfriendly', 'whispering'],
#     'en-US-SaraNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
#                          'unfriendly', 'whispering'],
#     'en-US-TonyNeural': ['angry', 'cheerful', 'excited', 'friendly', 'hopeful', 'sad', 'shouting', 'terrified',
#                          'unfriendly', 'whispering']
# }


# # Function to generate SSML for a given segment with specified style
# def generate_ssml(start_time, end_time, text, voice, style):
#     duration = end_time - start_time
#     words_per_minute = 160  # Average speaking rate
#     num_words = len(text.split())
#     actual_duration = (num_words / words_per_minute) * 60  # in seconds
#     speaking_rate = actual_duration / duration

#     ssml = f'''
#     <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
#         <voice name="{voice}">
#             <mstts:express-as style="{style}">
#                 <prosody rate="{speaking_rate:.2f}">{text}</prosody>
#             </mstts:express-as>
#         </voice>
#     </speak>
#     '''
#     return ssml


# # Function to synthesize speech from SSML
# def synthesize_ssml_to_speech(ssml):
#     synthesizer = SpeechSynthesizer(speech_config=speech_config)
#     result = synthesizer.speak_ssml_async(ssml).get()
#     audio_stream = AudioDataStream(result)
#     return audio_stream


# # Streamlit UI
# # st.title('Text-to-Speech Generator')
# #
# # # Input field for captions
# # captions_input = st.text_area('Enter the captions as JSON', height=200)
# #
# # # Dropdown menu for selecting voice
# # selected_voice = st.selectbox('Select Voice', list(voices_and_styles.keys()))
# #
# # # Dropdown menu for selecting style based on selected voice
# # selected_style = st.selectbox('Select Speaking Style', voices_and_styles[selected_voice])
# #
# # generate_button = st.button('Generate Speech')


# # Function to generate TTS for all segments
# def generate_tts_for_segments(captions, voice, style):
#     all_voice_list = []
#     for segment in captions:
#         start_time = segment['start']
#         end_time = segment['end']
#         text = segment['text']

#         ssml = generate_ssml(start_time, end_time, text, voice, style)
#         audio_stream = synthesize_ssml_to_speech(ssml)
#         audio_file = f'{int(time.time())}_speech_{start_time}_{end_time}.wav'  # Unique filename based on timestamp
#         audio_stream.save_to_wav_file(audio_file)
#         all_voice_list.append(audio_file)
#     audio_file = f'{int(time.time())}_speech_{start_time}_{end_time}.wav'
#     combine_audio_files(all_voice_list, audio_file)
#     return audio_file

# client = AzureOpenAI(
#             azure_endpoint="https://open-ai-east-us-2.openai.azure.com/",
#             api_key="777a11c72ed74d45aa8d8abf92c87d19",
#             api_version="2023-05-15")

# from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
# import os
# from moviepy.editor import TextClip, CompositeVideoClip, ColorClip
# import moviepy.config as mp_config
# from voice_extraction import Conv_TDF_net_trimm, KimVocal
# import torch
# from scipy.io.wavfile import write

# def remove_background_music(music_array, samplerate):

#     music_tensor = torch.tensor(music_array, dtype=torch.float32)

#     ONNX_MODEL_PATH = "Kim_Vocal.onnx"

#     model_raw_python = Conv_TDF_net_trimm(
#         model_path=ONNX_MODEL_PATH,
#         use_onnx=True,
#         target_name="vocals",
#         L=11,
#         l=3,
#         g=48,
#         bn=8,
#         bias=False,
#         dim_f=11,
#         dim_t=8,
#     )

#     kimvocal = KimVocal()
#     vocals_tensor = kimvocal.demix_vocals(
#         music_tensor=music_tensor,
#         sample_rate=samplerate,
#         model=model_raw_python,
#     )

#     return vocals_tensor.numpy()



# def add_caption_to_frame(frame, text, text_color, bg_color, is_bold, is_italic, horizontal_alignment):
#     height, width = frame.shape[:2]

#     # Choose font based on bold and italic settings
#     if is_bold and is_italic:
#         font = cv2.FONT_HERSHEY_TRIPLEX | cv2.FONT_ITALIC
#     elif is_bold:
#         font = cv2.FONT_HERSHEY_TRIPLEX
#     elif is_italic:
#         font = cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC
#     else:
#         font = cv2.FONT_HERSHEY_SIMPLEX

#     font_scale = max(width, height) / 1500 #1
#     thickness = max(1, int(font_scale * (2 if is_bold else 1))) #2 if is_bold else 1
#     line_type = cv2.LINE_AA

#     # Convert colors to tuples if they're not already
#     text_color = tuple(map(int, text_color))
#     bg_color = tuple(map(int, bg_color))

#     # Split text into multiple lines
#     max_width = int(width * 0.8)
#     words = text.split()
#     lines = []
#     current_line = []
#     current_width = 0

#     for word in words:
#         word_size = cv2.getTextSize(word + " ", font, font_scale, thickness)[0]
#         if current_width + word_size[0] <= max_width:
#             current_line.append(word)
#             current_width += word_size[0]
#         else:
#             lines.append(" ".join(current_line))
#             current_line = [word]
#             current_width = word_size[0]

#     if current_line:
#         lines.append(" ".join(current_line))

#     # Calculate text block dimensions
#     text_height = sum([cv2.getTextSize(line, font, font_scale, thickness)[0][1] for line in lines])
#     line_height = cv2.getTextSize("Tg", font, font_scale, thickness)[0][1]
#     padding = int(max(10, font_scale * 20)) #20
#     block_height = text_height + (len(lines) + 1) * padding
#     block_width = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines]) + 2 * padding

#     # Position the text block
#     y_position = int(height * 0.7) - block_height // 2

#     # Draw background rectangle covering the text region
#     x_start = (width - block_width) // 2 if horizontal_alignment == "center" else (
#         padding if horizontal_alignment == "left" else width - block_width - padding)
#     cv2.rectangle(frame, (x_start, y_position), (x_start + block_width, y_position + block_height), bg_color, -1)

#     # Draw text
#     for line in lines:
#         text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
#         if horizontal_alignment == "left":
#             x_position = x_start + padding
#         elif horizontal_alignment == "right":
#             x_position = width - text_size[0] - padding
#         else:  # center
#             x_position = (width - text_size[0]) // 2

#         y_position += line_height + padding
#         cv2.putText(frame, line, (x_position, y_position), font, font_scale, text_color, thickness, line_type)

#     return frame


# def process_frame(get_frame, t, text_color, bg_color, is_bold, is_italic, horizontal_alignment, captions):
#     frame = get_frame(t).astype(np.uint8).copy()
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
#     current_time = t

#     for caption in captions:
#         if caption["start"] <= current_time < caption["end"]:
#             frame = add_caption_to_frame(frame, caption["text"], text_color, bg_color, is_bold, is_italic,
#                                          horizontal_alignment)
#             break

#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert back to RGB
#     return frame

# def parse_captions(captions_string):
#     # Regular expression to match the pattern of each caption entry
#     pattern = re.compile(r'start:(\d+(\.\d+)?), end:(\d+(\.\d+)?), text: (.*?)(?=; start:|$)')
#     matches = pattern.findall(captions_string)

#     captions = []
#     for match in matches:
#         start = float(match[0])
#         end = float(match[2])
#         text = match[4].strip()
#         captions.append({"start": start, "end": end, "text": text})

#     return captions


# def create_video_with_text(text, duration, output_path):
#     os.environ["IMAGEMAGICK_BINARY"] = "/usr/bin/convert"

#     # Set up the video properties
#     width, height = 720, 1280  # Dimensions for the video
#     background_color = (0, 0, 0)  # White background
#     fontsize = 56  # Adjusted font size for the text
#     txt_color = 'white'  # Text color

#     # Create a background clip
#     background = ColorClip(size=(width, height), color=background_color, duration=duration)

#     # Create a text clip with the "Ultra-Regular" font
#     txt_clip = TextClip(text, fontsize=fontsize, color=txt_color, font='DejaVu-Sans-Bold', size=(width, height),
#                         method='caption', align='center').set_duration(duration)

#     # Overlay the text clip on the background clip
#     video = CompositeVideoClip([background, txt_clip])

#     silent_audio = AudioFileClip("silent_3_seconds.wav")
#     # Add the silent audio to the video
#     video = video.set_audio(silent_audio)

#     # Write the video to a file
#     video.write_videofile(output_path, fps=24, codec='libx264')
#     video.close()


# def replace_visuals_with_filler(main_video_path, filler_clips_data, output_path):
#     """
#     Replace visuals in a main video with filler clips while keeping the original audio.

#     Parameters:
#     - main_video_path (str): Path to the main video file.
#     - filler_clips_data (list): List of tuples, each containing:
#         - filler_clip_path (str): Path to a filler clip.
#         - start_time (float): Start time in seconds in the main video where the filler should begin.
#         - end_time (float): End time in seconds in the main video where the filler should end.
#     - output_path (str): Path to save the output video.

#     Example of filler_clips_data:
#     [
#         ("filler_clip1.mp4", 5, 10),  # filler clip from 5 to 10 seconds
#         ("filler_clip2.mp4", 20, 25)  # filler clip from 20 to 25 seconds
#     ]
#     """
#     # Load the main video
#     main_video = VideoFileClip(main_video_path)

#     # Create a list of segments (clips with or without filler)
#     segments = []

#     # Initialize the current time to start from the beginning of the main video
#     current_time = 0

#     for filler_clip_path, start_time, end_time in filler_clips_data:
#         # Add the original part of the video before the filler (if any)
#         if current_time < start_time:
#             original_segment = main_video.subclip(current_time, start_time)
#             segments.append(original_segment)

#         # Load the filler clip
#         filler_clip = VideoFileClip(filler_clip_path).subclip(0, end_time - start_time)

#         # Resize the filler clip to match the main video's resolution
#         filler_clip = filler_clip.resize(main_video.size)

#         # Ensure filler clip duration matches the defined range
#         filler_clip = filler_clip.set_duration(end_time - start_time)

#         # Add the filler clip
#         segments.append(filler_clip)

#         # Update the current time to the end of the filler
#         current_time = end_time

#     # Add the remaining part of the main video after the last filler (if any)
#     if current_time < main_video.duration:
#         remaining_segment = main_video.subclip(current_time, main_video.duration)
#         segments.append(remaining_segment)

#     # Concatenate all segments (with fillers and original clips)
#     final_video = concatenate_videoclips(segments)

#     # Add the original audio from the main video
#     final_video = final_video.set_audio(main_video.audio)

#     # Write the output video file
#     final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

#     # Close video clips to free up resources
#     main_video.close()
#     for clip in segments:
#         clip.close()

# import ast
# def ad_one_influencer_hero(user_requirement_dict, final_output, details_text, output_path, process_videos):
#     if process_videos:
#         one_influencer_hero_prompt = """You are the best video editor out there in the industry. You will be given a detailed metadata about segments that come from different source videos. The naming of segments is in the format: "<source_video_name>-<scene_number>.mp4" Your role is to select the best source video according to the user requirement. Then, in that source video you need to suggest best visual edits using segment from other videos. You have to output what part of the video can be replaced by visuals from segments coming from other source videos.
#         Output a python dictionary with the following two keys. The dict will be converted to JSON so format accordingly.
#         - source_video_name (str): Path to the best matching source video file. Take in all segmentsof it into consideration. For eg. If tmpshcg6vsr-0.mp4, tmpshcg6vsr-1.mp4, tmpshcg6vsr-2.mp4 best match the user requirements, output tmpshcg6vsr.mp4
#         - filler_clips_data (list): List of tuples, each containing:- filler_clip_path (str): Path to a filler clip. - start_time (float): Start time in seconds in the main video where the filler should begin. - end_time (float): End time in seconds in the main video where the filler should end. These segments should come from videos other than the source_video_name and decided using the matching visual aspects.
#         For eg. 
#             {source_video_name:"tmpshcg6vsr.mp4",
#         filler_clips_data:[("tmp0hklcbic-0.mp4", 5, 8), ("tmp0hklcbic-1.mp4", 20, 27)]}
#         Analyze all segments and output the source_video_name which is without the scene-number
#         Strictly output only the dict and nothing else. Don't give any exaplanation in the output. Make sure there are no newlines in the list inside the dictionary as it will be parsed on JSON.
#     """
#         metadata_prompt = """Here is the list of user requirements: \n""" + dict_to_string(user_requirement_dict) +  """\nHere is the list of available segments with metadata: \n""" + details_text + """\n\nHere is the output that the Ad Agent had generated: \n""" + final_output
#         response = client.chat.completions.create(
#             model="gpt-4o-global",
#             messages=[
#                 {"role": "system", "content": one_influencer_hero_prompt},
#                 {"role": "user", "content": metadata_prompt}
#             ],
#             temperature=0.5,
#         )
#         intermediate_data = response.choices[0].message.content

#         print(response.usage)
#         print(intermediate_data)
#         ispython = True if intermediate_data[:9] == "```python" else False
#         if ispython:
#             intermediate_data = intermediate_data[10:-3]
#         isjson = True if intermediate_data[:7] == "```json" else False
#         if isjson:
#             intermediate_data = intermediate_data[8:-3]
#         print(intermediate_data)
#         final_output = ast.literal_eval(intermediate_data)

#         replace_visuals_with_filler("/tmp/"+os.path.splitext(final_output["source_video_name"])[0], final_output["filler_clips_data"], output_path)
#     return True, {"updated_files":[], "updated_script":"", "summary":"", "final_quality_score":0.85}


# # Example usage
# # text = "Click the link below to get started"
# # duration = 10  # Duration in seconds
# # output_path = "output10_video.mp4"
# # create_video_with_text(text, duration, output_path)

# # mp_config.change_settings({"IMAGEMAGICK_BINARY": "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})

# # def remove_audio_noise(video_path, output_path):
# #     # Load video
# #     video = VideoFileClip(video_path)
# #     original_audio = video.audio
# #     audio_file_path = "mywav.wav"
# #     original_audio.write_audiofile(audio_file_path, codec='pcm_s16le')
# #     rate, data = wavfile.read("mywav.wav")
# #     print(rate)
# #     orig_shape = data.shape
# #     data = np.reshape(data, (2, -1))
# #     # perform noise reduction
# #     reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8, stationary=True, n_fft=512, hop_length=128,
# #                                     win_length=512, use_tqdm=False)
# #     wavfile.write("mywav_reduced_noise.wav", rate, reduced_noise.reshape(orig_shape))
# #     # Set the new audio to the video
# #     video = video.set_audio(AudioFileClip("mywav_reduced_noise.wav"))
# #
# #     # Write the output video file
# #     video.write_videofile(output_path, codec='libx264', audio_codec='aac')


# def overlay_captions(video_path, transcript, output_path):
#     # Load the video
#     video = VideoFileClip(video_path)

#     # Parse the transcript
#     captions = []
#     for line in transcript.split(";"):
#         parts = line.strip().split(", ")
#         start = float(parts[0].split(":")[1])
#         end = float(parts[1].split(":")[1])
#         text = parts[2].split(": ")[1]

#         # Create the text clip
#         text_clip = TextClip(text, fontsize=24, color='black', bg_color='white', size=(video.w, None), method='caption')
#         text_clip = text_clip.set_start(start).set_end(end).set_position(("center", "bottom"))

#         captions.append(text_clip)

#     # Create the final video with overlays
#     final_video = CompositeVideoClip([video] + captions)

#     # Write the result to a file
#     final_video.write_videofile(output_path, codec='libx264')


# # Example usage
# # video_path = "input_video.mp4"
# # transcript = "start:0, end:12.99, text: If you're tired of doing a lot of research into each game before you make your educated sports bet on it, let me tell you about an app called Odds R that can help you when making these decisions.; start:12.99, end:21.88, text: It was able to help me to figure out the best bets to make especially when I feel like going to look at the stats and see if this person will be playing better than this person, you know all that stuff.; start:21.88, end:30.75, text: The purpose of the app is to make sure that you get money and not lose it on just making wild guesses or wild bets.; start:30.75, end:39.62, text: This is an excellent app to help you win more and lose less. Odds R, check them out and stop wasting your time on all these devices trying to make sure that you are doing the best you can to make an educated guess on games."
# # output_path = "output_video.mp4"
# #
# # overlay_captions(video_path, transcript, output_path)


# def get_minimum_specs(video_paths):
#     min_fps = float('inf')
#     min_resolution = (float('inf'), float('inf'))

#     for path in video_paths:
#         clip = VideoFileClip(path)
#         min_fps = min(min_fps, clip.fps)
#         min_resolution = (min(min_resolution[0], clip.size[0]), min(min_resolution[1], clip.size[1]))
#         clip.close()

#     return min_fps, min_resolution

# def create_caption(text, start, end, video_width, video_height):
#     txt_clip = TextClip(text, fontsize=24, color='black', font='Arial', bg_color='white', align='center')
#     txt_clip = txt_clip.set_position(('center', video_height * 0.75)).set_duration(end - start).set_start(start)
#     return txt_clip


# def same_original_file(segment1, segment2):
#     def extract_original_name(segment):
#         match = re.match(r"^(.*)-(\d+)\.mp4$", segment)
#         if match:
#             return match.group(1)
#         else:
#             return "Error"

#     # Extract the original filenames from both segments
#     original1 = extract_original_name(segment1)
#     original2 = extract_original_name(segment2)

#     # Compare the original filenames
#     return original1 == original2

# def process_videos(video_paths, output_path, captions_flag, caption_data, background_music_array, ai_voiceover, selected_voice, selected_style, aspect_ratio, conversion_type, transition_duration=0):
#     import librosa
#     min_fps, min_resolution = get_minimum_specs(video_paths)
#     clips = []
#     converters = []

#     try:
#         for video_path in video_paths:
#             clip = VideoFileClip(video_path)

#             # Initialize the converter with the clip and the target aspect ratio
#             converter = VideoConverterWithBlur(clip, aspect_ratio)
#             converters.append(converter)

#             # Convert the clip using the converter instance
#             clip = converter.convert()

#             # Set the FPS of the clip
#             clip = clip.set_fps(min_fps)

#             # Append the converted clip to the clips list
#             clips.append(clip)

#         # Concatenate the video clips
#         final_clip = concatenate_videoclips(clips, method="compose")

#         if not ai_voiceover:
#             audio_clips = [clip.audio for clip in clips]
#             concatenated_audio = concatenate_audioclips(audio_clips)
#             concatenated_audio.write_audiofile(output_path+"original_audio_before_bgremove.wav")
#             music_array, sample_rate = librosa.load(output_path+"original_audio_before_bgremove.wav", mono=False, sr=44100)
#             output_audio_array = remove_background_music(music_array, sample_rate)
#             print(output_audio_array.shape)
#             original_audio_array = output_audio_array.T
#         else:
#             generated_audio = generate_tts_for_segments(caption_data, selected_voice, selected_style)
#             music_array, sample_rate = librosa.load(generated_audio, mono=False,
#                                                     sr=44100)
#             original_audio_array = music_array.T

#         video_duration = final_clip.duration
#         background_music_duration = len(background_music_array) / 44100

#         if background_music_duration < video_duration:
#             repeats = int(video_duration // background_music_duration) + 1
#             background_music_array = np.tile(background_music_array, (repeats, 1))
#         background_music_array = background_music_array[:len(original_audio_array)]

#         # Reduce the volume of the background music (adjust as needed)
#         background_music_array = background_music_array * 0.15

#         # Combine the original audio and background music
#         combined_audio_array = original_audio_array + background_music_array.astype(np.float32)

#         max_val = np.max(np.abs(combined_audio_array))
#         print(max_val)
#         if max_val > 1.0:
#             combined_audio_array = combined_audio_array / max_val

#         write(output_path+"combined_audio.wav", sample_rate, combined_audio_array)
#         audio_clip = AudioFileClip(output_path+"combined_audio.wav")

#         # Set the audio of the video clip
#         final_clip = final_clip.set_audio(audio_clip)

#         # Handle captions if needed
#         if captions_flag:
#             print("Adding captions")
#             print(type(final_clip))
#             final_clip_captions = final_clip.fl(lambda gf, t: process_frame(gf, t, (0, 0, 0), (255, 255, 255), True, False, "center", caption_data))
#             final_clip_captions.write_videofile(output_path, codec="libx264", audio_codec="aac")
#             final_clip_captions.close()
#         else:
#             final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

#     finally:
#         # Close the final clip and all converter objects to ensure resources are released
#         if 'final_clip' in locals():
#             final_clip.close()
        
#         for converter in converters:
#             converter.close()


# def generate_details_text(df):
#     details = []
#     for index, row in df.iterrows():
#         row_details = (
#             f"Filename: {row['filenames']}\n"
#             # f"Category: {row['categories']}\n"
#             f"Duration: {row['segment_durations']}\n"
#             f"Transcript: {row['transcript']}\n"
#             f"Scene Description: {row['sceneDescription']}\n"
#             # f"Age: {row['age']}\n"
#             f"Gender: {row['gender']}\n"
#             # f"Shot Location: {row['shotLocation']}\n"
#             f"Aesthetic Score: {row['aestheticScore']}\n"
#             f"Is there a Caption Overlay?: {row['isCaption']}\n"
#             # f"Audio Tone: {row['audio_tone']}\n"
#         )
#         details.append(row_details)
#     return "\n".join(details)

# # Generate and store the details text in a variable
# # details_text = generate_details_text(df)

# # Now details_text contains the formatted text for all rows
# # print(details_text)
# def dict_to_string(d):
#     return '\n'.join(f'{key}: {value}' for key, value in d.items())

# def get_themes(df):
#     import ast
#     response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": """You will get metadata about different Segments. You have to output themes of new Ads that can be generated by stitiching these segments. These themes should highlight product/lifestyle, scenes, important objects. The concept should be such that 3 varaiants of Ad with different hooks should be generated out of it. Each Ad is made up of atleast 3-4 segments. Based on the input segment metadata, show how many segments from the given data relate to a particular concept. Just output all the possible concepts in a python list format. Make sure to use double quotes in each element of the dict inside list and that elements are separated by comma. Make sure the themes are not redundant. Strictly don't output any description or segment filenames related to the themes.
#                  These are the possible values of narration style:
#                  One Influencer Hero: Ideal for a single, powerful influencer whose message aligns with the ad's objective. Strictly, it should have only one actor and all the segments should come from one source video. Segments without a person can come from any source video. Otherwise, don't output it as a category.
#                 Testimonial Style: Best when multiple influencers provide strong social proof through positive experiences.
#                 Lead with One, Transition to Others: Start with a compelling influencer and build a multi-faceted narrative with additional voices.
#                  For eg.
                 
#                  [{"Concept 1": "Embrace the Energy",
# "Core Idea": "A dynamic montage showcasing outdoor activities and the energizing effects of our protein bar.",
# "Key Visuals": "Running, yoga, nature, protein bar consumption, bright colors, energetic camera movements",
# "Recommended Music": "Upbeat music, nature sounds",
# "Recommended Emotional Tone": "voiceover emphasizing natural energy, Uplifting, empowering",
# "CTA": "Fuel your active life.",
# "Actors": "Female",
# "No. of segments related to this concept": "7",
# "Narration Style": "Testimonial Style"},
                 
# {"Concept 2": "Wholesome Fuel for Every Adventure"
# "Core Idea": "Highlight the protein bar as a healthy snack for outdoor adventures.",
# "Key Visuals": "Hiking, biking, picnics, natural settings, focus on the protein bar's portability",
# "Recommended Music": "Acoustic guitar music, nature sounds",
# "Recommended Emotional Tone": "friendly voiceover, Adventurous, lighthearted",
# "CTA": "Grab a bar and go!",
# "Actors": "Male, Female",
# "No. of segments related to this concept": "10"
# "Narration Style": "Lead with One, Transition to Others"}]
# """},
#                 {"role": "user", "content": "Here the metadata of segments: " +  generate_details_text(df)}
#             ],
#             temperature=0.5,
#         )
#     intermediate_data = response.choices[0].message.content
#     print(intermediate_data)
#     ispython = True if intermediate_data[:9]=="```python" else False
#     if ispython:
#         intermediate_data = intermediate_data[10:-3]
#     list_of_dicts = ast.literal_eval(intermediate_data)
#     list_of_strings = [str(d) for d in list_of_dicts]
#     return list_of_strings

# def no_of_related_segments(df, list_of_concepts):
#     import ast
#     response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": """You will get metadata about different Segments. You have to output the no. of segments that are related with each concept. Output concept title, no. of related segments and related segment filenames and scene description and transcript for the related segments.
# """},
#                 {"role": "user", "content": "Here are the concepts for which we need to find the no. of related segments: " + "\n".join(list_of_concepts) + "Here the metadata of segments: " +  generate_details_text(df)}
#             ],
#             temperature=0.5,
#         )
#     intermediate_data = response.choices[0].message.content
#     print(intermediate_data)
#     ispython = True if intermediate_data[:9]=="```python" else False
#     if ispython:
#         intermediate_data = intermediate_data[10:-3]
    
#     return intermediate_data

# def create_ad(df, user_requirement_dict):
#     transactional_prompt = """You are a pro video editor and an expert in making ads. You will be given bite sized segments with metadata and you will have to select the best segments that can be merged together to create perfect advertisements. 
#      You know the flow of events for Ads that are as follows:
#      Transactional Ad: Hook, Problem (Buildup), Solution, Social Proof.
#      Strictly adhere to the User requirements. The output should be a list of segment filenames that when concatenated will give the best cohesive output ad around 15s. IMPORTANT: The most important thing is that the final Ad should look complete and cohesive. It should flow and look like one complete Ad. It should not have abrupt ending. It is okay if you break the order of segment categories or the Final Ad is longer. You can output any no. files if needed to fill any gaps. Also, write the script (that will be converted to audio) for this Ad. You will output the filenames and transcript in a python dict format. Always Include segments from multiple files.
#      Example Output:
#      {
#         "files": ["file1.mp4", "file8.mp4", "file3.mp4"],
#         "Script": "This is a sample script",
#      }
#      Just output the dict and nothing else.
#     """
#     details_text = generate_details_text(df)
#     metadata_prompt = """Here is the list of user requirements: \n""" + dict_to_string(user_requirement_dict) + """\nHere is the list of available segments with metadata: \n""" + details_text
#     # print(transactional_prompt)
#     # print(metadata_prompt)
#     response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {"role": "system", "content": transactional_prompt},
#                     {"role": "user", "content": metadata_prompt}
#                 ],
#                 temperature=0.5,
#             )
#     intermediate_data = response.choices[0].message.content

#     print(response.usage)
#     ispython = True if intermediate_data[:9]=="```python" else False
#     if ispython:
#         intermediate_data = intermediate_data[10:-3]
#     print(intermediate_data)
#     final_output = intermediate_data #json.loads(intermediate_data)
#     # list_of_videos = eval(intermediate_data)
#     # process_videos(final_output["files"], "FinalTransactionalAd.mp4")
#     return user_requirement_dict ,final_output, details_text



# def ad_critic(user_requirement_dict, final_output, details_text, filename, cta_text, is_process_videos, iscaption, is_ai_voiceover, selected_voice, selected_style, aspect_ratio, conversion_type, m_value, temperature=0.5):
#     critic_prompt = """You are the best Ad critic out there in the business. The Ads that you finalize are used for Meta Ad Library for campaigns of Fortune 500 companies. So you make sure that it is of highest quality. I have an agent that works on creating Ads by merging bite sized segments. \
#                     I will provide the metadata it had, and the files it selected for making the best Ad. The loose structure of an Ad is: Hook, Problem (Buildup), Solution, Social Proof.
#                     You have to tell whether the given Ad is good to be given as the final output to the Fortune 500 company. If not, suggest changes or mention that an Ad is simply not possible out of the given segments. Make sure and take utmost care that the final Ad is cohesive and and the duration is according to the user's requirement.
#                     It is important to note that these segments come from different original files. The naming convention of segments is {original_filename}-{segment_number}.mp4. STRICTLY - Each Ad should be a combination of segments from multiple original files. If it's not possible to have segments from multiple files, write your own script.
#                     The output should be in a python dict format.
#                     Here is the definition of the output fields:
#                     "original_quality_score": A score between 0.0 to 1.0. The quality of the original Ad. 1.0 being the best.
#                     "critic" - Output "good" if the existing combination is perfect. Output "bad" if changes are required in the existing combination of segments. Output "none" if no good Ad is possible from the given segment content.
#                     "feedback": This is the feedback on the final Ad. Mention the changes that you will be making and the reasons for the same. Also mention how you will be using segments from different original files.
#                     "updated_files": Updated files list based on the above conditions.
#                     "updated_files_duration": Duration of the updated files in seconds.
#                     "updated_script": If the critic is "good", the updated_script should resemble the exact script from the existing segments. 
#                     If the critic is "bad" there can be two cases: 
#                     Case 1: Another combination of existing segments (using existing transcript) can be used to generate the best Ad. In this case, give the updated script (It should be a combination of transcripts from the original segments only. STRICTLY - DO NOT GENERATE AYTHING ON YOUR OWN). Also give the updated files that when merged together will create the best Ad. In this case, keep "is_script_from_segments" Yes.
#                     Case 2: No combination of existing transcripts make the best Ad OR The user hasn't liked the only possible combination OR All segments are coming from a single file. In this case write your own script (that will be converted to audio overlay), and update the files to select the best files that visually match the script (Use files information). IMPORTANT: Make sure to write script according to length of each segment. STRICTLY, the script should be around 2-2.5 words per second. Ensure same duration of the converted audio length to the final Ad duration from the files. In this case, keep "is_script_from_segments" No.
#                     If the critic is "none", keep the updated script empty.
#                     "is_script_from_segments": "Yes" or "No" based on the above conditions. Make sure it is in double quotes
#                     "summary": This is the summary of the final Ad by linking it with user's input.
#                     "keywords": Keywords that we can get from the final Ad (both audio and visual).
#                     "features": A list of audio and visual features. for eg ["Tone: Excited", "Female Speaking", "Location: Outdoors"]
#                     "final_quality_score": A score between 0.0 to 1.0. This is the score of the quality of the Ad after the changes.
#                      For eg.
#                     {
#                     "original_quality_score": 0.45,
#                     "critic": "good",
#                     "feedback": "This is the feedback", 
#                     "updated_files": ["file1.mp4", "file5.mp4", file2.mp4""] // If review of existing is good, output the existing filenames. If bad output the updated filenames, Make sure the files go with the entire script i.e. there should be files for each sentence of the script. If none output an empty list.
#                     "updated_files_duration": [3.53, 6.5, 2.0], //Duration of the updated files in seconds
#                     "updated_script": "start:0, end:5, text: This is the updated script line 1; start:5, end:10, text: This is the updated script line 2;", //The start and end time are the time of when the segment starts and ends in the Final Ad. The start time of any segment is where the last segment ends. The end time of any segment will be (last segment end + its duration) and should be coherent with the visual clips
#                     "is_script_from_segments": "Yes" //Yes if the entire script is from the existing segments, No if you've written even one sentence of script on your own. Write your own script only if really required.
#                     "summary": "This is the final Ad summary"
#                     "keywords": ["keyword1", "keyword2", "keyword3"],
#                     "features": ["feature1", "feature2", "feature3"],
#                     "final_quality_score": 0.75,
#                     }
#                     Output only the dict and nothing else. Don't put comma after the last element in the dict. The dict will be parsed as JSON so make sure all charcters are escaped correctly if required.
#                     Things to keep in mind:
#                     1. Audio Visual Synergy
#                     2. Maximize the final_quality_score (Ad Impact).
#                     3. The final Ad's duration should STRICTLY be according to the user's requirement. If it is around 15 seconds use 3 segments. If it is around 20 seconds use 4 segments. If it is around 30 seconds, use at least 6 segments. If it is around 45 seconds, use at least 9 segments. If it is around 60 seconds, use at least 13 segments. If it is around 90 seconds, use at least 20 segments. If it is around 120 seconds, use at least 25 segments to fulfil the requirement. Use segments without transcript to fill in if required.
#                     4. According to user's requirement, focus more on just the product or overall lifestyle that involves that product.
#                     5. Cohesiveness- The pieces should make sense one after other as a whole. It should flow and look like one complete Ad from start to end. Review the script and segments to ensure there is no logical cutoff. IMPORTANT: Make sure it has a good start and good closing. Always Include segments from multiple files.
#                     6. Strictly adhere to all the user requirements for creating the final Ad. If that's not possible, don't create an Ad.
#                     7. If the user requests a One Influencer Hero narrative style, Strictly, it should have only one actor and all the segments should come from one source video. Segments without a person can come from any source video.
#                     8. Don't use segments without any transcript for Ad Creation that are longer than 5 seconds."""
#     metadata_prompt = """Here is the list of user requirements: \n""" + dict_to_string(user_requirement_dict) +  """\nHere is the list of available segments with metadata: \n""" + details_text + """\n\nHere is the output that the Ad Agent had generated: \n""" + final_output
#     response = client.chat.completions.create(
#         model="gpt-4o-global",
#         messages=[
#             {"role": "system", "content": critic_prompt},
#             {"role": "user", "content": metadata_prompt}
#         ],
#         temperature=temperature,
#         # max_tokens=int(4000 * m_value)  # Adjust max_tokens based on m_value
#     )
#     intermediate_data = response.choices[0].message.content

#     print(response.usage)
#     # print(intermediate_data)
#     ispython = True if intermediate_data[:9] == "```python" else False
#     if ispython:
#         intermediate_data = intermediate_data[10:-3]
#     isjson = True if intermediate_data[:7] == "```json" else False
#     if isjson:
#         intermediate_data = intermediate_data[8:-3]
#     print(intermediate_data)
#     final_output = json.loads(intermediate_data)
    
#     if is_process_videos and final_output["is_script_from_segments"]=="No":
#         tts_prompt = """You will be given a string with segments of texts and the no. of words that each text should be converted to. Rewrite each text to meaningful quality sentences according to the no. of words given with it.
#         Here is the string:\n""" + final_output["updated_script"] + """\nHere is the word count per each segment: [""" + ', '.join(map(str, [round(x * m_value) for x in final_output["updated_files_duration"]])) + """]\nOutput just the rewrited string in the exact same format with the changed text. Ensure that there are no newlines in the string and colon even at the end of last text."""
        
        
#         print(tts_prompt)

#         response = client.chat.completions.create(
#             model="gpt-4o-global",
#             messages=[
#                 {"role": "system", "content": tts_prompt},
#             ],
#             temperature=0.5,
#         )
#         intermediate_data = response.choices[0].message.content

#         print(intermediate_data)
#         final_output["updated_script"] = intermediate_data

#     try:
#         if final_output["updated_files"]:
#             if is_process_videos:
#                 # filename = "FinalTransactionalAd" + str(int(temperature*10)) +".mp4"
#                 create_video_with_text(cta_text,3,"cta.mp4")
#                 # print(final_output["updated_files"])
#                 # print(final_output["updated_files"].append("cta.mp4"))
#                 final_output["updated_files"].append("cta.mp4")
#                 print(final_output["updated_files"])
#                 background_music_clip = AudioFileClip("bgmusic.mp3")
#                 bgarray = background_music_clip.to_soundarray(fps=44100)
#                 process_videos(final_output["updated_files"], filename, iscaption, parse_captions(final_output["updated_script"]), bgarray, is_ai_voiceover, selected_voice, selected_style, aspect_ratio, conversion_type)

#             # overlay_captions("FinalTransactionalAd.mp4", final_output["updated_script"], "FinalTransactionalAdCaption.mp4")
#             return True, final_output
#         else:
#             return False, False
#     except Exception as e:
#         print(e)
#         return False, False



# ##############################################

# voiceover_flag_status = False
# def status_voiceover_flag(voiceover_flag=False):
#     return voiceover_flag

# voiceover_flag_status = status_voiceover_flag()
# def eliminate_talking_head(df,voiceover_flag_status):
#     if voiceover_flag_status:
#         talking_head_rows = df[df['talkingHead'] == True]
#         return talking_head_rows
#     else:
#         return df


# ################################################

# if __name__ == "__main__":
#     df = pd.read_csv("segment_metadata2.csv")
#     requirements, output, details_text = create_ad(df, {"Ad Description": "An advertisemnt Build a bear"})
#     # requirements, output, details_text = create_ad(df, {"Selling Points of the product": "It gives the best odds fro betting", "Pain points of the customers": "It addresses the pain points of not getting the best odds and losing bets.", "Ad Description": "None", "Ad tone": "None", "Actor Gender Preference": "Any", "Shot Location Preference": "None"})
#     # requirements, output, details_text = create_ad(df, {"Selling Points of the product": """FlavCity provides easy-to-follow, healthy recipes and meal prep ideas""", "Pain points of the customers": """Some customers find the recipes to be time-consuming and requiring specific, sometimes hard-to-find ingredients, which can be inconvenient.""", "Ad Description": "I want to create a video that showcases all the different types of food items possible with their protein powder. Strictly focus on the variety of items. Keep it quick and don't show the entire recipe.", "Ad tone": "Any", "Actor Gender Preference": "Any", "Shot Location Preference": "Any"})
#     # ad_critic(requirements, output, details_text, 0.2)
#     q,w = ad_critic(requirements, output, details_text, "firstad.mp4","This is CTA",True, True, 0.5)
#     # a,b,c,d = ad_critic({"Ad Description": "An advertisemnt Build a bear", "Other Requirements": "An Ad was already made as a combination of the following files. Do not make a similar Ad. Make a different version now. Current Ad: " + ', '.join(r)}, output, details_text, 0.5)
#     a, b = ad_critic({"Ad Description": "An advertisement on Build a bear", "Other Requirements": "The user did not like the ad that was already created. Create an entirely new and imapctful version while making sure to use segments from multiple videos."}, "\nFilenames: " + ', '.join(w["updated_files"]) + "\n Script: " + w["updated_script"], details_text, "secondad.mp4","This is CTA",True, True,  0.5)
#     x, y = ad_critic({"Ad Description": "An advertisement on Build a bear", "Other Requirements": "The user did not like the ad that was already created. Create an entirely new and imapctful version while making sure to use segments from multiple videos."}, "\nFilenames: " + ', '.join(b["updated_files"]) + "\n Script: " + b["updated_script"], details_text, "thirdad.mp4","This is CTA",True, True, 0.5)

#     # ad_critic(requirements, output, details_text, 0.5)
#     # ad_critic(requirements, output, details_text, 0.9)
