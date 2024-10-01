from openai import OpenAI, AzureOpenAI
from moviepy.editor import VideoFileClip
import base64
from io import BytesIO
from typing import List
from PIL import Image
import json
# from transformers import pipeline
#
# emotion_classifier = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")
# print("Loaded Emotion Classifier")
#
# emotion_to_tone = {
#     "admiration": "Passionate",
#     "caring": "Passionate",
#     "love": "Passionate",
#     "gratitude": "Passionate",
#     "amusement": "Happy",
#     "joy": "Happy",
#     "pride": "Happy",
#     "excitement": "Happy",
#     "relief": "Happy",
#     "approval": "Happy",
#     "optimism": "Happy",
#     "anger": "Sad",
#     "annoyance": "Sad",
#     "disappointment": "Sad",
#     "disapproval": "Sad",
#     "disgust": "Sad",
#     "embarrassment": "Sad",
#     "fear": "Sad",
#     "grief": "Sad",
#     "nervousness": "Sad",
#     "remorse": "Sad",
#     "sadness": "Sad",
#     "curiosity": "Surprise",
#     "desire": "Surprise",
#     "insight": "Surprise",
#     "realization": "Surprise",
#     "neutral": "Neutral",
#     "surprise": "Surprise"
# }
#
# def get_audio_attributes(transcript):
#     """
#     Extract audio attributes such as tone and gender from the video.
#     """
#     emotion = emotion_classifier(transcript)
#     detected_emotion = emotion[0]["label"]
#     print(detected_emotion)
#
#     # Remove temporary audio file
#     # Determine the audio tone based on detected emotion
#     audio_tone = emotion_to_tone.get(detected_emotion, "Neutral")
#
#     return audio_tone
    # No audio detected in the video


client = AzureOpenAI(azure_endpoint = "https://open-ai-east-us-2.openai.azure.com/",
                     api_key="777a11c72ed74d45aa8d8abf92c87d19",
                     api_version="2023-05-15")

def summarize_transcript(transcript):
    prompt = f"Generate keywords and summary from the following text:\n{transcript}. Output it in a dict with keys keywords (whose value is a string of comma separated keywords) and summary (whose value is a text summary). Output only the dict and nothing else."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
                {"role": "system", "content": "Generate keywords and summary from the provided text."},
                {"role": "user", "content": prompt}
                 ])
    response_dict = response.choices[0].message.content
    print(response_dict)
    response_dict = json.loads(response_dict)
    print(response.usage)
    return response_dict["keywords"], response_dict["summary"]


def create_image_dicts(base64_images: List[str]) -> List[dict]:
    image_dicts = []

    for base64_image in base64_images:
        image_dict = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "low"
            }
        }
        image_dicts.append(image_dict)

    return image_dicts


def video_to_base64_frames(video_path: str) -> List[str]:
    # Open the video file
    video = VideoFileClip(video_path)

    # Get the duration of the video
    duration = video.duration

    # Define the specific timestamps (25% and 75% of the video duration)
    timestamps = [duration * 0.25, duration * 0.75]

    base64_frames = []

    for timestamp in timestamps:
        # Capture the frame at the given timestamp
        frame = video.get_frame(timestamp)

        # Convert the frame (numpy array) to an image
        img = Image.fromarray(frame)

        # Save the image to a BytesIO object
        buffered = BytesIO()
        img.save(buffered, format="JPEG")

        # Encode the BytesIO object as a base64 string
        frame_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Add the base64 string to the list
        base64_frames.append(frame_base64)

    return base64_frames


def analyze_video(video_path, transcript = None, use_gpt = True):

    # try:
    video_clip = VideoFileClip(video_path)
    has_audio = video_clip.audio is not None
    audio_tone, audio_gender, summary, keywords, segment_transcript = None, None, None, None, ""
    try:
        if use_gpt:
            audio_tone = "neutral" #get_audio_attributes(transcript)
            content_list = [
                        {"type": "text",
                         "text": """Make a dictionary (at video level) of scene description, age and gender of unique people, if there is a promo code or not, shot location (indoors or outdoors), summary and aesthetic score (out of 100) .The images below are keyframes of a video.
                         A sample output is as follows:
                         {
                         "sceneDescription": Text, //Description of the scene. If there is any person refer them as a male or female and not a person. Keep it in a exhaustive keyword format so that it is best suited for NLP search. It will be converted to embeddings for NLP Search.
                         "age": {"Person1":40, "Person2":20}, // Dict of individual ages, return empty dict if no person
                         "gender": {"Person1":"Male", "Person2":"Female"}, // , return empty dict if no person
                         "isPromoCode": False, //Boolean
                         "shotLocation": "Indoors", //Text
                         "summary": Text, //Visual Summary
                         "aestheticScore": 50,
                         "keywords": Text,
                         "isHumanExist": True //Boolean if a human is present or not in the frame
                         "isCaption": True //Boolean if a caption overlay is present in the frame
                         "talkingHead": True // Boolean if a person is close to the camera, even if:
                    // - They are slightly turned to the side (side profile),
                    // - They are looking down, left, or right but still positioned close or face front to the camera.
                    // Consider it a 'talking head' if the person is primarily focused in the frame, even if not facing the camera directly.
                         }
                         Output only the dict and nothing else.
                                 """}]
            frame_list = video_to_base64_frames(video_path)
            # print(frame_list)
            image_dicts_list = create_image_dicts(frame_list)
            # print(image_dicts_list)
            content_list.extend(image_dicts_list)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that gives visual attributes from am image!"},
                    {"role": "user", "content": content_list}
                ],
                temperature=0.0,
            )
            print(response.choices[0].message.content)
            print(response.usage)
            final_dict = eval(response.choices[0].message.content)
            print(type(final_dict["age"]))
            print(type(final_dict["gender"]))
            print(type(final_dict["isPromoCode"]))
            return [final_dict["sceneDescription"], final_dict["age"], final_dict["gender"], final_dict["isPromoCode"], final_dict["shotLocation"], final_dict["summary"], final_dict["aestheticScore"], audio_tone, final_dict["keywords"], final_dict["isHumanExist"], final_dict["isCaption"], final_dict["talkingHead"]]
        else:
            return [None, None, None, None, None, None, None, None, None, None, None, None]
    except Exception as e:
        print(e)
        return [None, None, None, None, None, None, None, None, None, None, None, None]
    finally:
        video_clip.close()


if __name__ == "__main__":
    # First video
    result1 = analyze_video("/home/purvangi/Downloads/video_17249410083306375-0.mp4")
    print("Result 1: ", result1)