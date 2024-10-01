# import streamlit as st
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
        # st.audio(audio_file, format='audio/wav')
        # st.success(f'Generated speech for segment: "{text}" saved to: {audio_file}')


# Generate speech for all segments if button is clicked
# if generate_button:
#     try:
#         captions = json.loads(captions_input)
#         if not selected_style or not selected_voice:
#             st.error('Please select a speaking style and a voice.')
#         else:
#             generate_tts_for_segments(captions, selected_voice, selected_style)
#     except json.JSONDecodeError:
#         st.error('Invalid JSON format. Please enter the captions as valid JSON.')