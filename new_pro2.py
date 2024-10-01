import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from segmentation import VideoTranscriptionProcessor
import tempfile
import pandas as pd
from ad_creation import create_ad, ad_critic, get_themes, no_of_related_segments, ad_one_influencer_hero,status_voiceover_flag
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
import time
import os
import json
from azure.cognitiveservices.speech import AudioDataStream, SpeechSynthesizer, SpeechConfig, SpeechSynthesisOutputFormat
import atexit

# Create a list to store temp files for cleanup
temp_files_to_clean = []



# Voice and supported styles dictionary
voices_and_styles = {
    'de-DE-ConradNeural': ['cheerful'],
    'en-GB-RyanNeural': ['chat', 'cheerful'],
    # ... (rest of the dictionary)
}

# Azure TTS subscription key and region
subscription_key = 'f74fe7fc6f6a48879e486a5b33e1653d'
region = 'westus'

# Configure the speech synthesis
speech_config = SpeechConfig(subscription=subscription_key, region=region)
speech_config.set_speech_synthesis_output_format(SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)

def cleanup_temp_files():
    # Cleanup all temp files
    for temp_file in temp_files_to_clean:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Temporary file {temp_file} deleted.")

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

# Register the cleanup function to be called when the program exits
atexit.register(cleanup_temp_files)

def extract_segments(transcript, word_indices):
    words = transcript.split()

    # Initialize a list to hold the segmented pieces
    segments = []
    print(word_indices)
    # Loop through the word indices to extract segments
    for segment in word_indices:
        # Extract start and end indices
        start_idx = segment['word']['start']
        end_idx = segment['word']['end']

        # Extract the corresponding segment from the transcript
        segment_text = " ".join(words[start_idx:end_idx+1])

        # Append the segment text to the list of segments
        segments.append(segment_text)

    return segments


def format_script(script):
    segments = script.split('; ')
    formatted_segments = []

    for segment in segments:
        if segment.strip():  # Check if the segment is not empty
            parts = segment.split(', ')
            start = parts[0].split(':')[1].strip()
            end = parts[1].split(':')[1].strip()
            text = parts[2].split(':')[1].strip()
            formatted_segments.append(f"Start: {start}, End: {end}\nText: {text}\n")

    formatted_script = "\n".join(formatted_segments)
    return formatted_script


def analyze_tone(df,index,row):
    from metadata import analyze_video

    transcript = None
    if "transcript" in row and row['transcript']:
        transcript = row['transcript']
    video_meta_data = analyze_video(row['filenames'], transcript)
    for col, value in zip(new_columns, video_meta_data):
            df.at[index, col] = value
    print(df)

st.title('Ad Generator')

uploaded_videos = st.file_uploader("Upload Videos", type=["mp4", "mov", "avi"], accept_multiple_files=True)

# Step 2: Process Videos and Generate Options
if st.button("Process Videos"):
    all_output_files, all_categories, all_segment_durations, all_transcripts = [], [], [], []
    if uploaded_videos:
        # Example: Assume each video generates a set of options
        st.session_state.options = []
        for uploaded_video in uploaded_videos:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_video.read())
                temp_file_path = temp_file.name
                # st.write(temp_file_path)
            st.write(uploaded_video.name)
            
            temp_files_to_clean.append(temp_file_path)

            processor = VideoTranscriptionProcessor(temp_file_path)
            output_files, categories, duration, transcript_text, word_index, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio = processor.segment_video()
            all_output_files.extend(output_files)
            all_categories.extend(categories)
            all_segment_durations.extend(segment_durations)
            if transcript_text and 'word' in word_index[0]:
                text_transcript = ''.join([word['word'] for word in transcript_text])
                print(text_transcript)
                all_transcripts.extend(extract_segments(text_transcript, word_index))
            else:
                all_transcripts.extend(["None"] * len(output_files))

            # if os.path.exists(temp_file_path):
            #     os.remove(temp_file_path)

        data = {
            'filenames': all_output_files,
            'categories': all_categories,
            'segment_durations': all_segment_durations,
            'transcript': all_transcripts,
        }

        # Creating the dataframe
        df = pd.DataFrame(data)
        print(df)

        df.to_csv("new_segment_data.csv", index=False)

        df = pd.read_csv("new_segment_data.csv")

        new_columns = ['sceneDescription', 'age', 'gender', 'isPromoCode', 'shotLocation', 'summary',
                        'aestheticScore', 'audio_tone', 'keywords', 'isHumanExist', 'isCaption']
        for col in new_columns:
            df[col] = None
        results = []
        start = time.process_time()

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = []
            for index, row in df.iterrows():
                future = pool.submit(analyze_tone, df,index,row )
                for future in as_completed(futures):
                    results.append(future.result())
                    print(results)
        print("Time take to generated tone", time.process_time() - start)


        start = time.process_time()

        df.to_csv("new_segment_metadata.csv", index=False)
        themes = get_themes(df)

        st.session_state.options.extend(themes)
        
        st.success("Videos processed! Please select from the options below.")
    else:
        st.error("Please upload at least one video to process.")

# Step 3: Display Options for Selection
import re
if 'options' in st.session_state and st.session_state.options:

    def format_list_of_strings(input_list):
        formatted_list = []
        
        for item in input_list:
            # Parse the string into a dictionary
            item = re.sub(r"(?<!\\)'", '"', item)
            data = json.loads(item)
            
            # Create a formatted string
            formatted_string = ""
            for key, value in data.items():
                formatted_string += f"{key}:\n    {value}\n\n"
            
            # Add the formatted string to the list
            formatted_list.append(formatted_string.strip())
        
        return formatted_list

    def process_strings(input_list):
        output_list = []
        for string in input_list:
            # Step 1: Remove first and last character
            if len(string) > 1:
                string = string[1:-1]
            else:
                string = ""
            
            # Step 2: Replace all single quotes with empty string
            string = string.replace("'", "")
            
            # Step 3: Insert a newline after each comma
            # string = string.replace(",", ",\n")
            
            # Add the processed string to the output list
            output_list.append(string)
        
        return output_list
    
    def parse_concept(concept_str):
        concept_dict = {}
        key = None
        value = []
        
        # Split the input by commas and loop through each part
        for part in concept_str.split(', '):
            if ':' in part:
                if key:
                    # Join the previous value parts and store in the dictionary
                    concept_dict[key] = ' '.join(value).strip()
                # Split into new key-value pair
                key, new_value = part.split(':', 1)
                key = key.strip()
                value = [new_value.strip()]
            else:
                # This part belongs to the previous value
                value.append(part.strip())
        
        # Don't forget to add the last key-value pair
        if key:
            concept_dict[key] = ' '.join(value).strip()
        
        return concept_dict
    
    def parse_concept2(concept_str):
        concept_dict = {}
        key = None
        value = []
        
        # Split the input by commas and loop through each part
        for part in concept_str.split(', '):
            if ':' in part:
                if key:
                    if key == 'Key Visuals':
                        concept_dict[key] = ', '.join(value).strip()
                    else:
                    # Join the previous value parts and store in the dictionary
                        concept_dict[key] = ' '.join(value).strip()
                # Split into new key-value pair
                key, new_value = part.split(':', 1)
                key = key.strip()
                value = [new_value.strip()]
            else:
                # This part belongs to the previous value
                value.append(part.strip())
        
        # Don't forget to add the last key-value pair
        if key:
            concept_dict[key] = ' '.join(value).strip()
        
        return concept_dict

    def format_concept(concept_dict):
        formatted_concept = ', '.join([f"{key}: {value}" for key, value in concept_dict.items()])
        return formatted_concept

    def extract_distinct_values(concept_list):
    # Initialize empty sets for each field
        key_visuals_set = set()
        recommended_music_set = set()
        recommended_emotional_tone_set = set()
        actors_set = set()
        narration_style_set = set()
        
        # Iterate through each concept in the list
        for concept_str in concept_list:
            # Parse the concept into a dictionary
            concept_dict = parse_concept2(concept_str)
            st.write(concept_dict)
            
            # Extract and split values by commas (if any) and add to the respective set
            
            if 'Key Visuals' in concept_dict:
                key_visuals_set.update([item.strip() for item in concept_dict['Key Visuals'].split(',')])
            
            if 'Recommended Music' in concept_dict:
                recommended_music_set.update([item.strip() for item in concept_dict['Recommended Music'].split(',')])
            
            if 'Recommended Emotional Tone' in concept_dict:
                recommended_emotional_tone_set.update([item.strip() for item in concept_dict['Recommended Emotional Tone'].split(',')])
            
            if 'Actors' in concept_dict:
                actors_set.update([item.strip() for item in concept_dict['Actors'].split(',')])
            
            if 'Narration Style' in concept_dict:
                narration_style_set.update([item.strip() for item in concept_dict['Narration Style'].split(',')])
        
        return {
            "Key Visuals": key_visuals_set,
            "Recommended Music": recommended_music_set,
            "Recommended Emotional Tone": recommended_emotional_tone_set,
            "Actors": actors_set,
            "Narration Style": narration_style_set
        }
    
    print(st.session_state.options)
    
    formatted_list = process_strings(st.session_state.options)
    st.write(formatted_list)
    all_values_set = extract_distinct_values(formatted_list)
    st.write(all_values_set)
    # st.write(all_values_set)
    
    # selected_options = st.radio("Choose a concept:", formatted_list)

    selected_items = {}

    st.title("Select upto 3 Concepts")

    # Loop through the list and create a checkbox for each item with a unique key
    for i, item in enumerate(formatted_list):
        selected_items[item] = st.checkbox(item, key=f"checkbox_{i}")

    # Store the selected items in a list
    selected_options = [item for item, selected in selected_items.items() if selected]

    edited_options = []

    for i, concept_str in enumerate(selected_options):
        st.write(f"### Edit Concept {i+1}")
        
        # Parse the concept string into a dictionary
        concept_dict = parse_concept2(concept_str)
        # st.write(concept_dict)
        # Create input fields for each key-value pair
        concept_dict[f"Concept {i+1}"] = st.text_input(f"Concept {i+1}:", value=concept_dict.get(next(k for k in concept_dict if k.startswith("Concept")), '')) #st.text_input(f"Concept {i+1}:", value=concept_dict.get(f"Concept {i+1}", ''))
        concept_dict['Core Idea'] = st.text_area(f"Core Idea {i+1}:", value=concept_dict.get('Core Idea', ''))
        # concept_dict['Key Visuals'] = st.text_area(f"Key Visuals {i+1}:", value=concept_dict.get('Key Visuals', ''))
        concept_dict['Key Visuals'] = st.multiselect(f"Key Visuals {i+1}:", options=list(all_values_set["Key Visuals"]), default=concept_dict.get('Key Visuals', '').split(', '))
        concept_dict['Recommended Music'] = st.multiselect(f"Recommended Music {i+1}:", options=list(all_values_set["Recommended Music"]), default=concept_dict.get('Recommended Music', '').split(', '))
        concept_dict['Recommended Emotional Tone'] = st.multiselect(f"Recommended Emotional Tone {i+1}:", options=list(all_values_set["Recommended Emotional Tone"]), default=concept_dict.get('Recommended Emotional Tone', '').split(', '))
        concept_dict['Actors'] = st.multiselect(f"Actors {i+1}:", options=list(all_values_set["Actors"]), default=concept_dict.get('Actors', '').split(', '))
        concept_dict['Narration Style'] = st.multiselect(f"Narration Style {i+1}:", options=list(all_values_set["Narration Style"]), default=concept_dict.get('Narration Style', '').split(', '))
        # concept_dict['Recommended Music'] = st.text_input(f"Recommended Music {i+1}:", value=concept_dict.get('Recommended Music', ''))
        # concept_dict['Recommended Emotional Tone'] = st.text_input(f"Recommended Emotional Tone {i+1}:", value=concept_dict.get('Recommended Emotional Tone', ''))
        # # concept_dict['CTA'] = st.text_input(f"CTA {i+1}:", value=concept_dict.get('CTA', ''))
        # concept_dict['Actors'] = st.text_input(f"Actors {i+1}:", value=concept_dict.get('Actors', ''))
        # # concept_dict['No. of segments related to this concept'] = st.text_input(f"No. of segments related to this concept {i+1}:", value=concept_dict.get('No. of segments related to this concept', ''))
        # concept_dict['Narration Style'] = st.text_input(f"Narration Style {i+1}:", value=concept_dict.get('Narration Style', ''))
        # Reformat the concept back into the original string format
        formatted_concept = format_concept(concept_dict)
        edited_options.append(formatted_concept)

    # Display the selected items list
    # st.write("Selected items:")
    # st.write(selected_options)

    st.write("### Configurations")
    # selected_options = st.multiselect("Select themes from processed videos:", st.session_state.options)
    ad_duration = st.number_input('Enter duration', min_value=0, max_value=500, value=20)
    gender = "Not specified"#st.selectbox("Select the gender:",("Male", "Female", "Any"))
    product_weightage = 50 #st.slider("Product Weightage (100 - x will be given to lifestyle)", 0, 100, 50)
    caption_in_video = "Not Specified"
    scene_tags = "Not Specified"
    selected_voice = "None"
    selected_style = "None"
    aspect_ratio_option = "9:16"
    conversion_option = "Blurred Background"
    cta_text = st.text_area('CTA Text:', value="Get started now!!")
    voiceover_flag = False
    voiceover_flag = st.checkbox("Generate AI Voiceover")

    ################################################

    status_voiceover_flag(voiceover_flag)
    ###############################################
    caption_flag = st.checkbox("Generate Caption")

    if 'sample_played' not in st.session_state:
        st.session_state.sample_played = False
    if 'last_voice' not in st.session_state:
        st.session_state.last_voice = None
    if 'last_style' not in st.session_state:
        st.session_state.last_style = None

    if voiceover_flag:
        selected_voice = st.selectbox('Select Voice', list(voices_and_styles.keys()))
        selected_style = st.selectbox('Select Speaking Style', voices_and_styles[selected_voice])
        
        if selected_voice != st.session_state.last_voice or selected_style != st.session_state.last_style:
            st.session_state.sample_played = False
            st.session_state.last_voice = selected_voice
            st.session_state.last_style = selected_style

        if not st.session_state.sample_played:
            sample_text = [
                {"start": 0, "end": 3, "text": "This is a sample to test the voiceover"}
            ]

            sample = generate_tts_for_segments(sample_text, selected_voice, selected_style)
            st.audio(sample)

            st.session_state.sample_played = True

    if st.button("Check Number of Related Segments"):
        df = pd.read_csv("new_segment_metadata.csv")
        output = no_of_related_segments(df, edited_options)
        st.write(output)

    # Add this near the other UI elements, before the "Create Ad" button
    speed_option = st.selectbox("Select Ad Generation Speed:", ["Fast", "Medium", "Slow"])    
    # Step 4: Final Processing Based on Selected Options
    if st.button("Create Ad"):
        if edited_options:
            print(edited_options)
            # Example: Final processing based on selected options
            # Replace this with your actual final processing logic
            df = pd.read_csv("new_segment_metadata.csv")
            requirements, output, details_text = create_ad(df, {"Ad Description": (edited_options[0]), "Ad Protagonist Gender": gender, "Scene Requirement Tags": scene_tags, "Focus on Product in Ad(%)": product_weightage, "Focus on Lifestyle in Ad(%)": 100 - product_weightage, "Include segments with inbuilt captions": caption_in_video, "Ad Duration:": ad_duration})
    # ad_created = ad_critic(requirements, output, details_text)
            ad_created_list, transcript_list, files_list = [], [], []

            col1, col2, col3 = st.columns(3)
            column_list = [col1, col2, col3]
            # temperature_list = [0.2, 0.5, 0.9]

            # # Add this near the other UI elements, before the "Create Ad" button
            # speed_option = st.selectbox("Select Ad Generation Speed:", ["Fast", "Medium", "Slow"])

            for i in range(5):
                output_file_name = "ad" + str(i) + ".mp4"
                if i < 2:
                    process_videos = False
                else:
                    process_videos = True
                
                # Convert speed option to m value
                if speed_option == "Fast":
                    m_value = 2.0
                elif speed_option == "Slow":
                    m_value = 1.0
                else:  # Medium
                    m_value = 1.5
                
                if i == 0:
                    ad_created, final_output = ad_critic(requirements, output, details_text, output_file_name, cta_text,
                                                         process_videos, caption_flag, voiceover_flag, selected_voice, selected_style, aspect_ratio_option, conversion_option, m_value)
                elif i<3:
                    concept_dict_new = parse_concept(edited_options[0])
                    narration_style = concept_dict_new.get('Narration Style', '')
                    print(narration_style)
                    if narration_style == """['One Influencer Hero']""":
                        ad_one_influencer_hero(requirements, "", details_text, output_file_name, process_videos)
                    else:
                        requirements[
                            "Other Requirements"] = "The user did not like the ad that was already created. Strictly change the hook (the part where the advertisement starts). Create an entirely new and imapctful version while making sure to use segments from multiple videos."
                        ad_created, final_output = ad_critic(requirements,
                                                            "\nFilenames: " + ', '.join(files_list[i - 1]) + "\n Script: " +
                                                            transcript_list[i - 1], details_text, output_file_name, cta_text,
                                                            process_videos, caption_flag, voiceover_flag, selected_voice, selected_style, aspect_ratio_option, conversion_option, m_value)
                elif i==3:
                    if len(edited_options)>1:
                        user_prompt = edited_options[1]
                        concept_dict_new = parse_concept(edited_options[1])
                    else:
                        user_prompt = edited_options[0]
                        concept_dict_new = parse_concept(edited_options[0])
                    narration_style = concept_dict_new.get('Narration Style', '')
                    print(narration_style)
                    if narration_style == """['One Influencer Hero']""":
                        ad_one_influencer_hero(requirements, "", details_text, output_file_name, process_videos)
                    else:
                        requirements[
                            "Other Requirements"] = "The user did not like the ad that was already created. Strictly change the hook (the part where the advertisement starts). Create an entirely new and imapctful version while making sure to use segments from multiple videos."
                        requirements["Ad Description"] = user_prompt
                        ad_created, final_output = ad_critic(requirements,
                                                            "\nFilenames: " + ', '.join(files_list[i - 1]) + "\n Script: " +
                                                            transcript_list[i - 1], details_text, output_file_name, cta_text,
                                                            process_videos, caption_flag, voiceover_flag, selected_voice, selected_style, aspect_ratio_option, conversion_option, m_value)
                
                else:
                    if len(edited_options)>2:
                        user_prompt = edited_options[2]
                        concept_dict_new = parse_concept(edited_options[2])
                    else:
                        user_prompt = edited_options[0]
                        concept_dict_new = parse_concept(edited_options[0])
                    narration_style = concept_dict_new.get('Narration Style', '')
                    print(narration_style)
                    if narration_style == """['One Influencer Hero']""":
                        ad_one_influencer_hero(requirements, "", details_text, output_file_name, process_videos)
                    else:
                        requirements[
                            "Other Requirements"] = "The user did not like both the ads that were already created. The start of the Ad (hook) you create should be strictly different from the start of both the previously create Ads. Create an entirely new and imapctful version while making sure to use segments from multiple videos."
                        requirements["Ad Description"] = user_prompt
                        ad_created, final_output = ad_critic(requirements,
                                                            "Ad 1: " + "\nFilenames: " + ', '.join(files_list[i - 1]) + "\n Script: " +
                                                            transcript_list[i - 1] + "\n\nAd 2: " + "\nFilenames: " + ', '.join(files_list[i - 2]) + "\n Script: " +
                                                            transcript_list[i - 2], details_text, output_file_name, cta_text,
                                                            process_videos, caption_flag, voiceover_flag, selected_voice, selected_style, aspect_ratio_option, conversion_option, m_value)
                files_list.append(final_output["updated_files"])
                transcript_list.append(final_output["updated_script"])
                if ad_created and process_videos:
                    with column_list[i - 2]:
                        st.write("Ad " + str(i - 1))
                        # if not generate_captions_flag:
                        #     st.video("ad"+str(i)+".mp4")
                        # else:
                        if final_output["final_quality_score"] <= 0.8:
                            st.warning("This Ad might be below the required quality standards.")
                        st.video("ad" + str(i) + ".mp4")
                        st.write("Summary:\n", final_output["summary"])
                        st.write("\n\nScript:\n")
                        st.write(format_script(final_output["updated_script"]))




            # final_result = f"Final result based on: {', '.join(selected_options)}"
            # st.success(final_result)
        else:
            st.error("Please select at least one option.")
