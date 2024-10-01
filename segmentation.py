# import traceback

# import pandas as pd
# from moviepy.editor import VideoFileClip
# import whisper
# from ad_prompt import ad_prompt
# import os
# from scenedetect import VideoManager, SceneManager
# from scenedetect.detectors import ContentDetector
# from scenedetect.stats_manager import StatsManager
# from openai import AzureOpenAI
# # import ffmpeg
# import shutil
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import numpy as np


# class VideoTranscriptionProcessor:
#     def __init__(self, video_path):
#         self.video_path = video_path
#         self.model = whisper.load_model("small.en")
#         self.client = AzureOpenAI(
#             azure_endpoint="https://open-ai-east-us-2.openai.azure.com/",
#             api_key="777a11c72ed74d45aa8d8abf92c87d19",
#             api_version="2023-05-15")

#     @staticmethod
#     def seconds_to_hh_mm_ss_ms(seconds):
#         hours = seconds // 3600
#         minutes = (seconds % 3600) // 60
#         seconds = seconds % 60
#         milliseconds = (seconds - int(seconds)) * 1000
#         return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"


#     def ensure_path_exists(self, path):
#         if not os.path.exists(path):
#             os.makedirs(path)
#             print(f"Path '{path}' created.")
#         else:
#             print(f"Path '{path}' already exists.")

#     def break_video_into_scenes(self, video_path, threshold=30.0, min_scene_len=60):
#         output_list, category_list, word_index_, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio,segment_thumbnails,segment_filenames =[],[], [], [], [], [], [], [], [], []
#         # Create the video manager
#         video_manager = VideoManager([video_path])
#         stats_manager = StatsManager()
#         scene_manager = SceneManager(stats_manager)

#         # Add ContentDetector to the scene manager with a custom threshold and minimum scene length
#         scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))

#         # base_name = os.path.basename(video_path)
#         # base_name, _ = os.path.splitext(base_name)

#         try:
#             # Start video_manager to get the frame rate, video length, etc.
#             video_manager.start()

#             # Perform scene detection
#             scene_manager.detect_scenes(frame_source=video_manager)
#             print("In Scene Detection")
#             # Get list of detected scenes
#             scene_list = scene_manager.get_scene_list(start_in_scene=True)
#             # scenes_info = []
#             print("scenes",scene_list)

#             # For each scene, extract the necessary information
#             for i, scene in enumerate(scene_list):
#                 start_time = scene[0].get_timecode()
#                 end_time = scene[1].get_timecode()

#                 # Calculate scene duration in seconds
#                 duration = scene[1].get_seconds() - scene[0].get_seconds()
#                 print("Duration of clip",duration)
#                 if duration ==0:
#                     continue

#                 file_name = os.path.basename(self.video_path)
#                 name_without_extension = file_name.rsplit('.', 1)[0]
#                 # Save each scene as a separate video file

#                 output_file = f'{name_without_extension}-{i}.mp4'
#                 # Save each scene as a separate video file
#                 output_base_path = "/".join(self.video_path.split("/")[:-1])
#                 output_file= output_base_path+"/".join(output_file.split("/")[:-1]) + '/segments/' + output_file.split("/")[-1]
#                 print("output file",output_file)
#                 ##########################################
#                 current_dir = os.path.dirname(os.path.abspath(__file__))
#                 segment_video_path = os.path.join(current_dir, output_file)

#                 # Create a backup of the original video
#                 backup_path = f"{segment_video_path}.backup"
#                 shutil.copy2(segment_video_path, backup_path)
#                 print(f"Processing video: {segment_video_path}")

#                 processed_video = process_video(segment_video_path)

#                 if processed_video:
#                     print(f"Video processing complete. Original file has been updated: {processed_video}")
#                     print("Note: The original video has been overwritten with the processed version.")
#                     print(f"A backup of the original video is available at: {backup_path}")
#                     os.remove(backup_path)
#                 else:
#                     print("No changes were made to the video.")
#                     os.remove(backup_path)  # Remove the backup if no changes were made

                
#                 print("Video cut Completed for ________________________________________:", output_file)

#                 # Reload the video clip to get the updated duration
#                 clip = VideoFileClip(segment_video_path)
#                 segment_thumbnail_path = "/".join(output_file.split("/")[:-1]) + '/thumbnails/' + \
#                                          output_file.split("/")[-1][:-3] + "jpg"
#                 self.ensure_path_exists("/".join(segment_thumbnail_path.split("/")[:-1]))
#                 clip.save_frame(segment_thumbnail_path, t=clip.duration / 2)

#                 audio_duration = clip.audio.duration  # Get the duration of the audio clip

#                 # Convert start_time and end_time to seconds
#                 start_time_seconds = pd.to_timedelta(start_time).total_seconds()
#                 end_time_seconds = pd.to_timedelta(end_time).total_seconds()

#                 # Ensure the time range is within the audio duration
#                 if start_time_seconds >= audio_duration:
#                     print(f"Skipping segment {i} as start_time_seconds ({start_time_seconds}) >= audio_duration ({audio_duration})")
#                     continue
#                 if end_time_seconds > audio_duration:
#                     end_time_seconds = audio_duration

#                 # Access the audio within the valid time range
#                 audio_segment = clip.audio.subclip(start_time_seconds, end_time_seconds)
#                 audio_path = f"{name_without_extension}-{i}.mp3"
#                 audio_segment.write_audiofile(audio_path)   

#                 height = clip.size[1]
#                 width = clip.size[0]





#                 ##########################################
#                 # clip = VideoFileClip(video_path).subclip(start_time, end_time)
#                 # segment_thumbnail_path = "/".join(output_file.split("/")[:-1]) + '/thumbnails/' + \
#                 #                          output_file.split("/")[-1][:-3] + "jpg"
#                 # self.ensure_path_exists("/".join(segment_thumbnail_path.split("/")[:-1]))
#                 # clip.save_frame(segment_thumbnail_path, t=clip.duration / 2)
#                 # ffmpeg.input(video_path, ss=start_time, to=end_time).output(output_file).run()
#                 # clip.write_videofile(output_file)
#                 # height = clip.size[1]
#                 # width = clip.size[0]

#                 # Extract a thumbnail

#                 # Append to lists
#                 output_list.append(output_file)
#                 category_list.append("others")
#                 word_index_.append({})
#                 segment_filenames.append(output_file.split("/")[-1])
#                 segment_durations.append(duration)
#                 segment_timestamps.append({"start": start_time_seconds, "end": end_time_seconds})

#                 # segment_timestamps.append({"start":pd.to_timedelta(start_time).total_seconds(), "end":pd.to_timedelta(end_time).total_seconds()})
#                 segment_height.append(height)
#                 segment_width.append(width)
#                 segment_thumbnails.append(segment_thumbnail_path)

#                 segment_is_audio.append(False)

#             return output_list, category_list, word_index_, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio,segment_thumbnails,segment_filenames

#         finally:
#             video_manager.release()


#     def get_transcription_and_sentences(self, video_clip):
#         import time

#         start = time.process_time()
#         duration = video_clip.duration
#         if not video_clip.audio:
#             return None, None, duration, None, None
#         audio_path = "temp.mp3"
#         video_clip.audio.write_audiofile(audio_path)

#         # def alter_df(df, duration):
#         #     last_index = df.index[-1]

#         #     # Adjust the "end" time for all rows except the last one
#         #     df['end'][:-1] = (0.3*df['end'][:-1]) + (0.7*df['start'].shift(-1)[:-1])

#         #     # For the last row, set the "end" time to the minimum of (end + 1) and duration
#         #     df.at[last_index, 'end'] = min(df.at[last_index, 'end'] + 1, duration)
#         #     df['end'] = df['end'].round(2)
#         #     # df.at[last_index, 'end'] = (df.at[last_index, 'end'] + duration) / 2
#         #     # df.at[last_index, 'end'] = df.at[last_index, 'end']
#         #     return df

#         def alter_df(df, duration):
#             last_index = df.index[-1]

#     # Calculate the two values for all rows except the last one
#             value1 = (0.2 * df['end'][:-1]) + (0.8 * df['start'].shift(-1)[:-1])
#             value2 = df['end'][:-1] + 0.5

#     # Set 'end' to the minimum of value1 and value2 for all rows except the last one
#             df['end'][:-1] = np.minimum(value1, value2)

#             last_value = df.at[last_index, 'end']
#             last_value1 = (0.5 * last_value) + (0.5 * duration)  # First value for last row
#             last_value2 = last_value + 0.5                          # Second value for last row

#             df.at[last_index, 'end'] = min(last_value1, last_value2)
#     # For the last row, set the "end" time to the minimum of (end + 1) and duration
#             # df.at[last_index, 'end'] = min(df.at[last_index, 'end'] + last_increase_factor, duration)

#     # Round the 'end' column to 2 decimal places
#             df['end'] = df['end'].round(2)

#             return df


#         # video_clip.close()
#         try:
#             result = self.model.transcribe(audio_path, word_timestamps=True)
#             word_list = [word for segment in result['segments'] for word in segment['words']]
#             print("This is the real word list from Whisper: ", word_list)
#             print("Time taken to generated transcript and sentences in function get_transcription_and_sentences:::", time.process_time() - start)

#             word_df = pd.DataFrame(word_list)
#             word_df.to_csv("original_data.csv", index=False)
#             print(word_df)
#             word_df = alter_df(word_df, duration)
#             word_df.to_csv("altered_data.csv", index=False)
#             print(word_df)
#         except Exception as e:
#             traceback.print_exc()
#             return None, None, duration, None, None
#         print("No. of words: ", len(word_list))
#         if len(word_list) < 5:
#             return None, None, duration, None, None
#         transcript__json = word_df.to_dict(orient='records')
#         try:
#             sentence_df = self.create_sentence_df(word_df)
#         except:
#             return None, None, duration, None, None
#         sentence_df = sentence_df.dropna()
#         print("Sentence DF original: ")
#         print(sentence_df)
#         sentence_df = sentence_df[sentence_df['end'] > sentence_df['start']]
#         sentence_df = sentence_df[sentence_df['end'] - sentence_df['start'] >= 1]
#         sentence_df = sentence_df.reset_index(drop=True)
#         print("Sentence DF new: ")
#         print(sentence_df)
#         os.remove(audio_path)  # Clean up the audio file
#         # os.remove(temp_video_path)

#         final_text = ""
#         for index, row in sentence_df.iterrows():
#             final_text += f"Statement: {row['sentence']}\nStart Time: {self.seconds_to_hh_mm_ss_ms(row['start'])}\nEnd Time: {self.seconds_to_hh_mm_ss_ms(row['end'])}\n"
#             print(final_text)
#         return final_text, sentence_df, duration, transcript__json, word_df


#     def create_sentence_df(self, word_df):
#         text_data = "\n".join([f"{row['word']} {row['start']} {row['end']}" for index, row in word_df.iterrows()])
#         categorization_prompt = """Your role is convert a transcript from word level to sentence level. You will be given the words, their start time and end time from a video's transcript. Your role is to output meaningful sentences. Output the sentence, sentence start time, sentence end time in a python list format. Each element in the list should be dictionary with keys sentence, start and end. Output only the list and nothing else."""

#         prompt2 = """The output sentences should STRICTLY be not less than 1 seconds or more than 10 seonds. If a sentence is less than 1s, merge it with another sentence. If a sentence is more than 10s, break it into 2 meaningful sentences. Output at least once sentence from the text you get. Here is the transcript input: """ + str(
#             text_data)

#         response = self.client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": categorization_prompt},
#                 {"role": "user", "content": prompt2 + text_data}
#             ],
#             temperature=0.0,
#         )
#         intermediate_data = response.choices[0].message.content

#         print(response.usage)
#         ispython = True if intermediate_data[:9]=="```python" else False
#         if ispython:
#             intermediate_data = intermediate_data[10:-3]
#         print(intermediate_data)
#         return pd.DataFrame(eval(intermediate_data))

#     def get_gpt_response(self, transcript):
#         categorization_prompt = ad_prompt
#         prompt2 = f"Now Give me the output for the following input:\nThe transcript for the video is: {transcript}"
#         response = self.client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": categorization_prompt},
#                 {"role": "user", "content": prompt2}
#             ],
#             temperature=0.0,
#         )
#         try:
#             print(response.choices[0].message.content)
#             print(response.usage)
#             response_content = response.choices[0].message.content
#             ispython = True if response_content[:9] == "```python" else False
#             if ispython:
#                 response_content = response_content[10:-3]
#             categorized_segments = eval(response_content)  # Safely evaluate the response
#             print("Categorized Segments", categorized_segments)
#             if not isinstance(categorized_segments, list) or not all(
#                     isinstance(seg, dict) for seg in categorized_segments):
#                 print("Invalid segment format received from GPT.")
#                 return []
#         except (SyntaxError, TypeError) as e:
#             print(f"Error parsing GPT response: {e}")
#             return []

#         return categorized_segments
    
#     def detect_all_scenes(self, video_path, threshold=50.0):
#         video_manager = VideoManager([video_path])
#         scene_manager = SceneManager()
#         scene_manager.add_detector(ContentDetector(threshold=threshold))

#         video_manager.set_downscale_factor(2)  # Faster processing at the cost of lower accuracy
#         video_manager.start()

#         scene_manager.detect_scenes(frame_source=video_manager)
#         scene_list = scene_manager.get_scene_list()

#         return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

# # Main function to check for scene changes and trim the video if necessary
#     def smart_trim_video(self, output_file):
#         # Load the video file
#         clip = VideoFileClip(output_file)
        
#         # Detect all scene changes in the video
#         scene_changes = self.detect_all_scenes(output_file)
        
#         # Check for scene changes in the first 0.3 seconds
#         start_trim = 0
#         for (start, end) in scene_changes:
#             if end <= 0.5:  # If a scene ends within the first 0.3 seconds
#                 start_trim = end  # Set the start trim to the end of that scene
#             else:
#                 break  # Stop once we find a scene change after the 0.3 seconds window

#         # Check for scene changes in the last 0.5 seconds
#         end_trim = clip.duration
#         for (start, end) in reversed(scene_changes):
#             if start >= clip.duration - 0.5:  # If a scene starts within the last 0.5 seconds
#                 end_trim = start  # Set the end trim to the start of that scene
#             else:
#                 break  # Stop once we find a scene change before the last 0.5 seconds window

#         # Trim the clip if necessary
#         if start_trim > 0 or end_trim < clip.duration:
#             trimmed_clip = clip.subclip(start_trim, end_trim)

#             # Save the trimmed clip to a temporary file first to avoid OS errors
#             temp_output_file = "temp_output_file.mp4"         
#             try:
#                 trimmed_clip.write_videofile(temp_output_file, codec="libx264", audio_codec="aac")
#                 # Replace the original file with the trimmed version
#                 os.replace(temp_output_file, output_file)
#                 print(f"Video trimmed and saved as {output_file}")
#             except Exception as e:
#                 print(e)
#             finally:
#                 # Ensure the temporary file is removed in case of an error
#                 if os.path.exists(temp_output_file):
#                     os.remove(temp_output_file)
#         else:
#             print("No scene changes detected in the first 0.3s or last 0.5s, video remains unchanged.")

#     def generate_clip(self,row,word_df_,row_index,output_list,category_list,word_index_,segment_durations,segment_timestamps,segment_height,segment_width,segment_is_audio):
#         video_clip = VideoFileClip(self.video_path)
#         try:

#             time_interval = row['time_delta'].replace(" - ", "-")
#             start_time, end_time = time_interval.split("-")
#             category = row['category'].replace(" ", "").lower()
#             start_time_seconds = pd.to_timedelta(start_time).total_seconds()
#             word_df_['start'] = word_df_['start'].round(2)
#             word_df_['end'] = word_df_['end'].round(2)
#             print(word_df_['start'][:20], start_time_seconds)
#             print(word_df_['start'][0], type(word_df_['start'][0]), start_time_seconds, type(start_time_seconds),
#                   word_df_['start'][0] == round(start_time_seconds, 2))
#             start_index = word_df_[word_df_['start'] == round(start_time_seconds, 2)].index[0]
#             end_time_seconds = pd.to_timedelta(end_time).total_seconds()
#             print(word_df_['end'][:20])
#             print(round(end_time_seconds, 2))
#             end_index = word_df_[word_df_['end'] == round(end_time_seconds, 2)].index[0]
#             print(start_time_seconds, end_time_seconds)
#             index = {"word": {"start": start_index, "end": end_index}}
#             print(index)
#             file_name = os.path.basename(self.video_path)
#             name_without_extension = file_name.rsplit('.', 1)[0]
#             # Save each scene as a separate video file
#             output_file = f'{name_without_extension}-{row_index}.mp4'
#             print(min(end_time_seconds, video_clip.duration))
#             subclip = video_clip.subclip(max(start_time_seconds-0.02,0), min(end_time_seconds, video_clip.duration))
#             try:
#                 # subclip = video_clip.subclip(max(start_time_seconds-0.02,0), min(end_time_seconds, video_clip.duration))
#                 # clip_ = VideoFileClip(self.video_path)
#                 width, height = subclip.size
#                 # Check if audio is present
#                 has_audio = subclip.audio is not None
#                 segment_duration = subclip.duration
#                 # output_file = "/".join(output_file.split("/")[:-1]) + '/segments/' + output_file.split("/")[-1]
#                 # self.ensure_path_exists("/".join(output_file.split("/")[:-1]))
#                 # segment_thumbnail_path = "/".join(output_file.split("/")[:-1]) + '/thumbnails/' + output_file.split("/")[-1][:-3] + "jpg"
#                 # self.ensure_path_exists("/".join(segment_thumbnail_path.split("/")[:-1]))
#                 # subclip.save_frame(segment_thumbnail_path, t=subclip.duration / 2)
#                 start_write_time = time.process_time()
#                 print("Writing Subclip")
#                 subclip.write_videofile(output_file)
#                 # self.smart_trim_video(output_file)
#                 print("SubClip Written")
#             except Exception as e:
#                 print(e)
#             finally:
#                 subclip.close()
#             # ffmpeg.input(self.video_path, ss=start_time_seconds, to=end_time_seconds+ 0.35).output(output_file).run()
#             print("Time taken to write clip to outputfile", time.process_time() - start_write_time)

#             # segment_thumbnails.append(segment_thumbnail_path)
#             segment_durations.append(segment_duration)
#             # segment_filenames.append(output_file.split("/")[-1])
#             # self.upload_file_azure(output_file.name, output_file, self.container_name)
#             output_list.append(output_file)
#             category_list.append(category)
#             word_index_.append(index)
#             segment_timestamps.append({"start": start_time_seconds, "end": end_time_seconds})
#             segment_height.append(height)
#             segment_width.append(width)
#             segment_is_audio.append(has_audio)
#             return True
#         finally:
#             video_clip.close()

#     def segment_video(self):
#         video_clip = VideoFileClip(self.video_path)
#         import time

#         start = time.process_time()
#         transcript, _, duration_, transcript_json_, word_df_ = self.get_transcription_and_sentences(video_clip)
#         print("Time take to generated transcript and sentences", time.process_time() - start)
#         start = time.process_time()

#         gpt_response = self.get_gpt_response(transcript)
#         print(f"Time take to generated gpt response for transcript   {time.process_time() - start}")

#         # print(len(word_df_))
#         if not isinstance(word_df_, pd.DataFrame) or len(word_df_) < 5 or (isinstance(gpt_response, list) and not gpt_response):
#             output_list, category_list, word_index_, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio = self.break_video_into_scenes(self.video_path)
#             return output_list, category_list, duration_, transcript_json_, word_index_, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio
#         output_list, category_list, word_index_, segment_thumbnails, segment_durations, segment_filenames, segment_timestamps, segment_height, segment_width, segment_is_audio = [], [], [], [], [], [], [], [], [], []

#         start = time.process_time()
#         num_processes = 1  # Number of threads to use
#         results = []
#         from concurrent.futures import ProcessPoolExecutor, as_completed

#         with ThreadPoolExecutor(max_workers=num_processes) as pool:
#             # output_list, category_list,word_index_, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio

#             futures = {pool.submit(self.generate_clip, row, word_df_, gpt_response.index(row), output_list,
#                                    category_list, word_index_, segment_durations,segment_timestamps, segment_height, segment_width,
#                                    segment_is_audio): row for row in gpt_response}
#             for future in as_completed(futures):
#                 results.append(future.result())
#             print(results)

#         """with ProcessPoolExecutor(max_workers=num_processes) as pool:
#             # output_list, category_list,word_index_, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio

#             futures = {pool.submit(self.generate_clip, row,word_df_,gpt_response.index(row) ,video_clip ,output_list,category_list,word_index_,segment_timestamps,segment_height,segment_width,segment_is_audio): row for row in gpt_response}
#             for future in as_completed(futures):
#                 results.append(future.result())
#             print(results)"""

#         """for row in gpt_response:
#             if row:
#                 time_interval = row['time_delta'].replace(" - ", "-")
#                 start_time, end_time = time_interval.split("-")
#                 category = row['category'].replace(" ", "").lower()
#                 start_time_seconds = pd.to_timedelta(start_time).total_seconds()
#                 word_df_['start'] = word_df_['start'].round(2)
#                 word_df_['end'] = word_df_['end'].round(2)
#                 print(word_df_['start'][:20], start_time_seconds)
#                 print(word_df_['start'][0], type(word_df_['start'][0]), start_time_seconds, type(start_time_seconds), word_df_['start'][0]==round(start_time_seconds,2))
#                 start_index = word_df_[word_df_['start'] == round(start_time_seconds,2)].index[0]
#                 end_time_seconds = pd.to_timedelta(end_time).total_seconds()
#                 end_index = word_df_[word_df_['end'] == round(end_time_seconds,2)].index[0]
#                 print(start_time_seconds, end_time_seconds)
#                 index = {"word":{"start":start_index, "end":end_index}}
#                 print(index)
#                 file_name = os.path.basename(self.video_path)
#                 name_without_extension = file_name.rsplit('.', 1)[0]
#                 # Save each scene as a separate video file
#                 output_file = f'{name_without_extension}-{gpt_response.index(row)}.mp4'

#                 # clip_ = VideoFileClip(self.video_path)
#                 subclip = video_clip.subclip(start_time_seconds, min(end_time_seconds + 0.35, video_clip.duration))
#                 width, height = subclip.size
#                 # Check if audio is present
#                 has_audio = subclip.audio is not None
#                 segment_duration = subclip.duration
#                 # output_file = "/".join(output_file.split("/")[:-1]) + '/segments/' + output_file.split("/")[-1]
#                 # self.ensure_path_exists("/".join(output_file.split("/")[:-1]))
#                 # segment_thumbnail_path = "/".join(output_file.split("/")[:-1]) + '/thumbnails/' + output_file.split("/")[-1][:-3] + "jpg"
#                 # self.ensure_path_exists("/".join(segment_thumbnail_path.split("/")[:-1]))
#                 # subclip.save_frame(segment_thumbnail_path, t=subclip.duration / 2)
#                 start_write_time = time.process_time()

#                 subclip.write_videofile(output_file)
#                 # ffmpeg.input(self.video_path, ss=start_time_seconds, to=end_time_seconds+ 0.35).output(output_file).run()
#                 print("Time taken to write clip to outputfile", time.process_time() - start_write_time)

#                 # segment_thumbnails.append(segment_thumbnail_path)
#                 segment_durations.append(segment_duration)
#                 # segment_filenames.append(output_file.split("/")[-1])
#                 # self.upload_file_azure(output_file.name, output_file, self.container_name)
#                 output_list.append(output_file)
#                 category_list.append(category)
#                 word_index_.append(index)
#                 segment_timestamps.append({"start":start_time_seconds, "end":end_time_seconds})
#                 segment_height.append(height)
#                 segment_width.append(width)
#                 segment_is_audio.append(has_audio)"""
#         # os.remove(temp_video_path)
#         print("Time taken to generate segments and write output", time.process_time() - start)

#         return output_list, category_list, duration_, transcript_json_, word_index_, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio



# if __name__ == "__main__":
#     import time
#     start = time.time()
#     processor = VideoTranscriptionProcessor("DeAnn Williams Reel Sept 27.mp4")
#     output_files, categories, duration, transcript_text, word_index, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio = processor.segment_video()
#     print(output_files)
#     print(duration)
#     print(transcript_text)
#     print(type(transcript_text))
#     print(word_index)
#     print(segment_durations)
#     print(segment_timestamps)
#     print(segment_height)
#     print(segment_width)
#     print(segment_is_audio)
#     print(time.time() - start)


import traceback

import pandas as pd
from moviepy.editor import VideoFileClip
import whisper
from ad_prompt import ad_prompt
import os
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.stats_manager import StatsManager
from openai import AzureOpenAI
# import ffmpeg
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


class VideoTranscriptionProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.model = whisper.load_model("small.en")
        self.client = AzureOpenAI(
            azure_endpoint="https://open-ai-east-us-2.openai.azure.com/",
            api_key="777a11c72ed74d45aa8d8abf92c87d19",
            api_version="2023-05-15")

    @staticmethod
    def seconds_to_hh_mm_ss_ms(seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        milliseconds = (seconds - int(seconds)) * 1000
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"


    def ensure_path_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Path '{path}' created.")
        else:
            print(f"Path '{path}' already exists.")

    def break_video_into_scenes(self, video_path, threshold=30.0, min_scene_len=60):
        output_list, category_list, word_index_, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio = [], [], [], [], [], [], [], []
        # Create the video manager
        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)

        # Add ContentDetector to the scene manager with a custom threshold and minimum scene length
        scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))

        # base_name = os.path.basename(video_path)
        # base_name, _ = os.path.splitext(base_name)

        try:
            # Start video_manager to get the frame rate, video length, etc.
            video_manager.start()

            # Perform scene detection
            scene_manager.detect_scenes(frame_source=video_manager)
            print("In Scene Detection")
            # Get list of detected scenes
            scene_list = scene_manager.get_scene_list(start_in_scene=True)
            # scenes_info = []
            print(scene_list)

            # For each scene, extract the necessary information
            for i, scene in enumerate(scene_list):
                start_time = scene[0].get_timecode()
                end_time = scene[1].get_timecode()

                # Calculate scene duration in seconds
                duration = scene[1].get_seconds() - scene[0].get_seconds()
                file_name = os.path.basename(self.video_path)
                name_without_extension = file_name.rsplit('.', 1)[0]
                # Save each scene as a separate video file

                output_file = f'{name_without_extension}-{i}.mp4'

                clip = VideoFileClip(video_path).subclip(start_time, end_time)
                # ffmpeg.input(video_path, ss=start_time, to=end_time).output(output_file).run()
                clip.write_videofile(output_file)
                height = clip.size[1]
                width = clip.size[0]

                # Extract a thumbnail

                # Append to lists
                output_list.append(output_file)
                category_list.append("others")
                word_index_.append({})
                segment_durations.append(duration)
                segment_timestamps.append({"start":pd.to_timedelta(start_time).total_seconds(), "end":pd.to_timedelta(end_time).total_seconds()})
                segment_height.append(height)
                segment_width.append(width)
                segment_is_audio.append(False)

            return output_list, category_list, word_index_, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio

        finally:
            video_manager.release()


    def get_transcription_and_sentences(self, video_clip):
        import time

        start = time.process_time()
        duration = video_clip.duration
        if not video_clip.audio:
            return None, None, duration, None, None
        audio_path = "temp.mp3"
        video_clip.audio.write_audiofile(audio_path)

        # def alter_df(df, duration):
        #     last_index = df.index[-1]

        #     # Adjust the "end" time for all rows except the last one
        #     df['end'][:-1] = (0.3*df['end'][:-1]) + (0.7*df['start'].shift(-1)[:-1])

        #     # For the last row, set the "end" time to the minimum of (end + 1) and duration
        #     df.at[last_index, 'end'] = min(df.at[last_index, 'end'] + 1, duration)
        #     df['end'] = df['end'].round(2)
        #     # df.at[last_index, 'end'] = (df.at[last_index, 'end'] + duration) / 2
        #     # df.at[last_index, 'end'] = df.at[last_index, 'end']
        #     return df

        def alter_df(df, duration):
            last_index = df.index[-1]

    # Calculate the two values for all rows except the last one
            value1 = (0.2 * df['end'][:-1]) + (0.8 * df['start'].shift(-1)[:-1])
            value2 = df['end'][:-1] + 0.5

    # Set 'end' to the minimum of value1 and value2 for all rows except the last one
            df['end'][:-1] = np.minimum(value1, value2)

            last_value = df.at[last_index, 'end']
            last_value1 = (0.5 * last_value) + (0.5 * duration)  # First value for last row
            last_value2 = last_value + 0.5                          # Second value for last row

            df.at[last_index, 'end'] = min(last_value1, last_value2)
    # For the last row, set the "end" time to the minimum of (end + 1) and duration
            # df.at[last_index, 'end'] = min(df.at[last_index, 'end'] + last_increase_factor, duration)

    # Round the 'end' column to 2 decimal places
            df['end'] = df['end'].round(2)

            return df


        # video_clip.close()
        try:
            result = self.model.transcribe(audio_path, word_timestamps=True)
            word_list = [word for segment in result['segments'] for word in segment['words']]
            print("This is the real word list from Whisper: ", word_list)
            print("Time taken to generated transcript and sentences in function get_transcription_and_sentences:::", time.process_time() - start)

            word_df = pd.DataFrame(word_list)
            word_df.to_csv("original_data.csv", index=False)
            print(word_df)
            word_df = alter_df(word_df, duration)
            word_df.to_csv("altered_data.csv", index=False)
            print(word_df)
        except Exception as e:
            traceback.print_exc()
            return None, None, duration, None, None
        print("No. of words: ", len(word_list))
        if len(word_list) < 5:
            return None, None, duration, None, None
        transcript__json = word_df.to_dict(orient='records')
        try:
            sentence_df = self.create_sentence_df(word_df)
        except:
            return None, None, duration, None, None
        sentence_df = sentence_df.dropna()
        print("Sentence DF original: ")
        print(sentence_df)
        sentence_df = sentence_df[sentence_df['end'] > sentence_df['start']]
        sentence_df = sentence_df[sentence_df['end'] - sentence_df['start'] >= 1]
        sentence_df = sentence_df.reset_index(drop=True)
        print("Sentence DF new: ")
        print(sentence_df)
        os.remove(audio_path)  # Clean up the audio file
        # os.remove(temp_video_path)

        final_text = ""
        for index, row in sentence_df.iterrows():
            final_text += f"Statement: {row['sentence']}\nStart Time: {self.seconds_to_hh_mm_ss_ms(row['start'])}\nEnd Time: {self.seconds_to_hh_mm_ss_ms(row['end'])}\n"
            print(final_text)
        return final_text, sentence_df, duration, transcript__json, word_df


    def create_sentence_df(self, word_df):
        text_data = "\n".join([f"{row['word']} {row['start']} {row['end']}" for index, row in word_df.iterrows()])
        categorization_prompt = """Your role is convert a transcript from word level to sentence level. You will be given the words, their start time and end time from a video's transcript. Your role is to output meaningful sentences. Output the sentence, sentence start time, sentence end time in a python list format. Each element in the list should be dictionary with keys sentence, start and end. Output only the list and nothing else."""

        prompt2 = """The output sentences should STRICTLY be not less than 1 seconds or more than 10 seonds. If a sentence is less than 1s, merge it with another sentence. If a sentence is more than 10s, break it into 2 meaningful sentences. Output at least once sentence from the text you get. Here is the transcript input: """ + str(
            text_data)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": categorization_prompt},
                {"role": "user", "content": prompt2 + text_data}
            ],
            temperature=0.0,
        )
        intermediate_data = response.choices[0].message.content

        print(response.usage)
        ispython = True if intermediate_data[:9]=="```python" else False
        if ispython:
            intermediate_data = intermediate_data[10:-3]
        print(intermediate_data)
        return pd.DataFrame(eval(intermediate_data))

    def get_gpt_response(self, transcript):
        categorization_prompt = ad_prompt
        prompt2 = f"Now Give me the output for the following input:\nThe transcript for the video is: {transcript}"
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": categorization_prompt},
                {"role": "user", "content": prompt2}
            ],
            temperature=0.0,
        )
        try:
            print(response.choices[0].message.content)
            print(response.usage)
            response_content = response.choices[0].message.content
            ispython = True if response_content[:9] == "```python" else False
            if ispython:
                response_content = response_content[10:-3]
            categorized_segments = eval(response_content)  # Safely evaluate the response
            print("Categorized Segments", categorized_segments)
            if not isinstance(categorized_segments, list) or not all(
                    isinstance(seg, dict) for seg in categorized_segments):
                print("Invalid segment format received from GPT.")
                return []
        except (SyntaxError, TypeError) as e:
            print(f"Error parsing GPT response: {e}")
            return []

        return categorized_segments
    
    def detect_all_scenes(self, video_path, threshold=50.0):
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        video_manager.set_downscale_factor(2)  # Faster processing at the cost of lower accuracy
        video_manager.start()

        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

# Main function to check for scene changes and trim the video if necessary
    def smart_trim_video(self, output_file):
        # Load the video file
        clip = VideoFileClip(output_file)
        
        # Detect all scene changes in the video
        scene_changes = self.detect_all_scenes(output_file)
        
        # Check for scene changes in the first 0.3 seconds
        start_trim = 0
        for (start, end) in scene_changes:
            if end <= 0.5:  # If a scene ends within the first 0.3 seconds
                start_trim = end  # Set the start trim to the end of that scene
            else:
                break  # Stop once we find a scene change after the 0.3 seconds window

        # Check for scene changes in the last 0.5 seconds
        end_trim = clip.duration
        for (start, end) in reversed(scene_changes):
            if start >= clip.duration - 0.5:  # If a scene starts within the last 0.5 seconds
                end_trim = start  # Set the end trim to the start of that scene
            else:
                break  # Stop once we find a scene change before the last 0.5 seconds window

        # Trim the clip if necessary
        if start_trim > 0 or end_trim < clip.duration:
            trimmed_clip = clip.subclip(start_trim, end_trim)

            # Save the trimmed clip to a temporary file first to avoid OS errors
            temp_output_file = "temp_output_file.mp4"         
            try:
                trimmed_clip.write_videofile(temp_output_file, codec="libx264", audio_codec="aac")
                # Replace the original file with the trimmed version
                os.replace(temp_output_file, output_file)
                print(f"Video trimmed and saved as {output_file}")
            except Exception as e:
                print(e)
            finally:
                # Ensure the temporary file is removed in case of an error
                if os.path.exists(temp_output_file):
                    os.remove(temp_output_file)
        else:
            print("No scene changes detected in the first 0.3s or last 0.5s, video remains unchanged.")

    def generate_clip(self,row,word_df_,row_index,output_list,category_list,word_index_,segment_durations,segment_timestamps,segment_height,segment_width,segment_is_audio):
        video_clip = VideoFileClip(self.video_path)
        try:

            time_interval = row['time_delta'].replace(" - ", "-")
            start_time, end_time = time_interval.split("-")
            category = row['category'].replace(" ", "").lower()
            start_time_seconds = pd.to_timedelta(start_time).total_seconds()
            word_df_['start'] = word_df_['start'].round(2)
            word_df_['end'] = word_df_['end'].round(2)
            print(word_df_['start'][:20], start_time_seconds)
            print(word_df_['start'][0], type(word_df_['start'][0]), start_time_seconds, type(start_time_seconds),
                  word_df_['start'][0] == round(start_time_seconds, 2))
            start_index = word_df_[word_df_['start'] == round(start_time_seconds, 2)].index[0]
            end_time_seconds = pd.to_timedelta(end_time).total_seconds()
            print(word_df_['end'][:20])
            print(round(end_time_seconds, 2))
            end_index = word_df_[word_df_['end'] == round(end_time_seconds, 2)].index[0]
            print(start_time_seconds, end_time_seconds)
            index = {"word": {"start": start_index, "end": end_index}}
            print(index)
            file_name = os.path.basename(self.video_path)
            name_without_extension = file_name.rsplit('.', 1)[0]
            # Save each scene as a separate video file
            output_file = f'{name_without_extension}-{row_index}.mp4'
            print(min(end_time_seconds, video_clip.duration))
            subclip = video_clip.subclip(max(start_time_seconds-0.02,0), min(end_time_seconds, video_clip.duration))
            try:
                # subclip = video_clip.subclip(max(start_time_seconds-0.02,0), min(end_time_seconds, video_clip.duration))
                # clip_ = VideoFileClip(self.video_path)
                width, height = subclip.size
                # Check if audio is present
                has_audio = subclip.audio is not None
                segment_duration = subclip.duration
                # output_file = "/".join(output_file.split("/")[:-1]) + '/segments/' + output_file.split("/")[-1]
                # self.ensure_path_exists("/".join(output_file.split("/")[:-1]))
                # segment_thumbnail_path = "/".join(output_file.split("/")[:-1]) + '/thumbnails/' + output_file.split("/")[-1][:-3] + "jpg"
                # self.ensure_path_exists("/".join(segment_thumbnail_path.split("/")[:-1]))
                # subclip.save_frame(segment_thumbnail_path, t=subclip.duration / 2)
                start_write_time = time.process_time()
                print("Writing Subclip")
                subclip.write_videofile(output_file)
                # self.smart_trim_video(output_file)
                print("SubClip Written")
            except Exception as e:
                print(e)
            finally:
                subclip.close()
            # ffmpeg.input(self.video_path, ss=start_time_seconds, to=end_time_seconds+ 0.35).output(output_file).run()
            print("Time taken to write clip to outputfile", time.process_time() - start_write_time)

            # segment_thumbnails.append(segment_thumbnail_path)
            segment_durations.append(segment_duration)
            # segment_filenames.append(output_file.split("/")[-1])
            # self.upload_file_azure(output_file.name, output_file, self.container_name)
            output_list.append(output_file)
            category_list.append(category)
            word_index_.append(index)
            segment_timestamps.append({"start": start_time_seconds, "end": end_time_seconds})
            segment_height.append(height)
            segment_width.append(width)
            segment_is_audio.append(has_audio)
            return True
        finally:
            video_clip.close()

    def segment_video(self):
        video_clip = VideoFileClip(self.video_path)
        import time

        start = time.process_time()
        transcript, _, duration_, transcript_json_, word_df_ = self.get_transcription_and_sentences(video_clip)
        print("Time take to generated transcript and sentences", time.process_time() - start)
        start = time.process_time()

        gpt_response = self.get_gpt_response(transcript)
        print(f"Time take to generated gpt response for transcript   {time.process_time() - start}")

        # print(len(word_df_))
        if not isinstance(word_df_, pd.DataFrame) or len(word_df_) < 5 or (isinstance(gpt_response, list) and not gpt_response):
            output_list, category_list, word_index_, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio = self.break_video_into_scenes(self.video_path)
            return output_list, category_list, duration_, transcript_json_, word_index_, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio
        output_list, category_list, word_index_, segment_thumbnails, segment_durations, segment_filenames, segment_timestamps, segment_height, segment_width, segment_is_audio = [], [], [], [], [], [], [], [], [], []

        start = time.process_time()
        num_processes = 1  # Number of threads to use
        results = []
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=num_processes) as pool:
            # output_list, category_list,word_index_, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio

            futures = {pool.submit(self.generate_clip, row, word_df_, gpt_response.index(row), output_list,
                                   category_list, word_index_, segment_durations,segment_timestamps, segment_height, segment_width,
                                   segment_is_audio): row for row in gpt_response}
            for future in as_completed(futures):
                results.append(future.result())
            print(results)

        """with ProcessPoolExecutor(max_workers=num_processes) as pool:
            # output_list, category_list,word_index_, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio

            futures = {pool.submit(self.generate_clip, row,word_df_,gpt_response.index(row) ,video_clip ,output_list,category_list,word_index_,segment_timestamps,segment_height,segment_width,segment_is_audio): row for row in gpt_response}
            for future in as_completed(futures):
                results.append(future.result())
            print(results)"""

        """for row in gpt_response:
            if row:
                time_interval = row['time_delta'].replace(" - ", "-")
                start_time, end_time = time_interval.split("-")
                category = row['category'].replace(" ", "").lower()
                start_time_seconds = pd.to_timedelta(start_time).total_seconds()
                word_df_['start'] = word_df_['start'].round(2)
                word_df_['end'] = word_df_['end'].round(2)
                print(word_df_['start'][:20], start_time_seconds)
                print(word_df_['start'][0], type(word_df_['start'][0]), start_time_seconds, type(start_time_seconds), word_df_['start'][0]==round(start_time_seconds,2))
                start_index = word_df_[word_df_['start'] == round(start_time_seconds,2)].index[0]
                end_time_seconds = pd.to_timedelta(end_time).total_seconds()
                end_index = word_df_[word_df_['end'] == round(end_time_seconds,2)].index[0]
                print(start_time_seconds, end_time_seconds)
                index = {"word":{"start":start_index, "end":end_index}}
                print(index)
                file_name = os.path.basename(self.video_path)
                name_without_extension = file_name.rsplit('.', 1)[0]
                # Save each scene as a separate video file
                output_file = f'{name_without_extension}-{gpt_response.index(row)}.mp4'

                # clip_ = VideoFileClip(self.video_path)
                subclip = video_clip.subclip(start_time_seconds, min(end_time_seconds + 0.35, video_clip.duration))
                width, height = subclip.size
                # Check if audio is present
                has_audio = subclip.audio is not None
                segment_duration = subclip.duration
                # output_file = "/".join(output_file.split("/")[:-1]) + '/segments/' + output_file.split("/")[-1]
                # self.ensure_path_exists("/".join(output_file.split("/")[:-1]))
                # segment_thumbnail_path = "/".join(output_file.split("/")[:-1]) + '/thumbnails/' + output_file.split("/")[-1][:-3] + "jpg"
                # self.ensure_path_exists("/".join(segment_thumbnail_path.split("/")[:-1]))
                # subclip.save_frame(segment_thumbnail_path, t=subclip.duration / 2)
                start_write_time = time.process_time()

                subclip.write_videofile(output_file)
                # ffmpeg.input(self.video_path, ss=start_time_seconds, to=end_time_seconds+ 0.35).output(output_file).run()
                print("Time taken to write clip to outputfile", time.process_time() - start_write_time)

                # segment_thumbnails.append(segment_thumbnail_path)
                segment_durations.append(segment_duration)
                # segment_filenames.append(output_file.split("/")[-1])
                # self.upload_file_azure(output_file.name, output_file, self.container_name)
                output_list.append(output_file)
                category_list.append(category)
                word_index_.append(index)
                segment_timestamps.append({"start":start_time_seconds, "end":end_time_seconds})
                segment_height.append(height)
                segment_width.append(width)
                segment_is_audio.append(has_audio)"""
        # os.remove(temp_video_path)
        print("Time taken to generate segments and write output", time.process_time() - start)

        return output_list, category_list, duration_, transcript_json_, word_index_, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio



if __name__ == "__main__":
    import time
    start = time.time()
    processor = VideoTranscriptionProcessor("/home/jay/Desktop/Cipio/vm_streamlit/video_17259065537835443.mp4")
    output_files, categories, duration, transcript_text, word_index, segment_durations, segment_timestamps, segment_height, segment_width, segment_is_audio = processor.segment_video()
    print(output_files)
    print(duration)
    print(transcript_text)
    print(type(transcript_text))
    print(word_index)
    print(segment_durations)
    print(segment_timestamps)
    print(segment_height)
    print(segment_width)
    print(segment_is_audio)
    print(time.time() - start)