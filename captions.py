import cv2
import numpy as np
from moviepy.editor import VideoFileClip
# import subprocess
#
#
# # Function to get available fonts using ImageMagick
# def get_available_fonts():
#     result = subprocess.run(['convert', '-list', 'font'], stdout=subprocess.PIPE)
#     fonts = result.stdout.decode('utf-8').split('\n')
#     font_dict = {}
#     current_font = None
#     for line in fonts:
#         if "Font:" in line:
#             current_font = line.split(": ")[1].strip()
#         if "glyphs:" in line and current_font:
#             font_dict[current_font] = line.split(": ")[1].strip()
#     return font_dict
#
#
# # Get available fonts
# available_fonts = get_available_fonts()
#
# # Display available fonts for user selection
# print("Available fonts:")
# for font in available_fonts.keys():
#     print(font)
#
# # Dictionary to map color names to BGR tuples (OpenCV uses BGR)
# color_name_to_bgr = {
#     "black": (0, 0, 0),
#     "white": (255, 255, 255),
#     "red": (0, 0, 255),  # BGR format
#     "green": (0, 255, 0),
#     "blue": (255, 0, 0),  # BGR format
#     "yellow": (0, 255, 255),
#     "cyan": (255, 255, 0),
#     "magenta": (255, 0, 255),
#     "gray": (128, 128, 128),
#     "grey": (128, 128, 128),
#     "orange": (0, 165, 255),
#     "purple": (128, 0, 128),
#     "pink": (203, 192, 255),
#     "brown": (42, 42, 165),
# }

# Get user inputs
# selected_font = input("Select Font: ")
# text_color_name = input("Enter text color (name, e.g., black, white, red): ").lower()
# bg_color_name = input("Enter background color (name, e.g., black, white, red): ").lower()
# is_bold = input("Bold text? (yes/no): ").lower() == "yes"
# is_italic = input("Italic text? (yes/no): ").lower() == "yes"
# horizontal_alignment = input("Horizontal alignment (left/center/right): ").lower()

# Convert color names to BGR tuples
# text_color = (0, 0, 0)  # Default to black if not found
# bg_color = (255, 255, 255)  # Default to white if not found

# print(f"Text color (BGR): {text_color}")
# print(f"Background color (BGR): {bg_color}")

# Captions data
# captions = [
#     {"start": 0, "end": 4.29, "text": "Amina is so ecstatic about her bear from Bill DeBear."},
#     {"start": 4.29, "end": 10.8,
#      "text": "This is what Amina chose for her bear, Lena, so that they can be matching vesties."},
#     {"start": 10.8, "end": 13.23, "text": "She gives her lots of hugs and kisses."},
#     {"start": 13.23, "end": 16.1, "text": "And now our new friend is finally ready to come home with us."}
# ]


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


# Input video path
# input_video_path = "/home/purvangi/Desktop/captions/FinalTransactionalAd1.mp4"
#
# # Output video path
# output_video_path = "/home/purvangi/Desktop/captions/output_video_caption29.mp4"
#
# # Load the video
# video = VideoFileClip(input_video_path)
#
# # Create the final video with user inputs
# final_video = video.fl(
#     lambda gf, t: process_frame(gf, t, text_color, bg_color, is_bold, is_italic, horizontal_alignment))
#
# # Write the output video
# final_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
#
# print(f"Video processing complete. Output saved to {output_video_path}")