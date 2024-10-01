from moviepy.editor import VideoFileClip, vfx, ImageClip, CompositeVideoClip
from PIL import Image, ImageFilter
import numpy as np
# from file_utils import save_and_get_video_details

def convert_with_cropping(clip, target_aspect_ratio):
    """
    Convert video to a new aspect ratio by cropping.

    Parameters:
    input_video (str): Path to the input video file.
    target_aspect_ratio (str): Desired aspect ratio ("16:9", "9:16", "1:1").

    Returns:
    tuple: Path to the output video, new dimensions, and file size.
    """
    # clip = VideoFileClip(input_video)
    width, height = clip.size

    if target_aspect_ratio == "16:9":
        new_height = width * 9 / 16
        new_clip = clip.crop(y_center=height/2, height=new_height)
    elif target_aspect_ratio == "9:16":
        new_width = height * 9 / 16
        new_clip = clip.crop(x_center=width/2, width=new_width)
    elif target_aspect_ratio == "1:1":
        new_side = min(width, height)
        new_clip = clip.crop(x_center=width/2, y_center=height/2, width=new_side, height=new_side)
    else:
        raise ValueError(f"Unsupported aspect ratio: {target_aspect_ratio}")

    output_path = f"output_cropped_{target_aspect_ratio.replace(':', '_')}.mp4"
    return new_clip

def convert_with_padding(clip, target_aspect_ratio):
    """
    Convert video to a new aspect ratio by adding black bars.

    Parameters:
    input_video (str): Path to the input video file.
    target_aspect_ratio (str): Desired aspect ratio ("16:9", "9:16", "1:1").

    Returns:
    tuple: Path to the output video, new dimensions, and file size.
    """
    # clip = VideoFileClip(input_video)
    width, height = clip.size

    # Determine new size and padding
    if target_aspect_ratio == "16:9":
        new_clip = clip.fx(vfx.resize, height=height)
        final_clip = new_clip.on_color(size=(int(new_clip.h * 16 / 9), new_clip.h), color=(0, 0, 0), col_opacity=1)
    elif target_aspect_ratio == "9:16":
        new_clip = clip.fx(vfx.resize, width=width)
        final_clip = new_clip.on_color(size=(new_clip.w, int(new_clip.w * 16 / 9)), color=(0, 0, 0), col_opacity=1)
    elif target_aspect_ratio == "1:1":
        if width / height > 1:  # 16:9 video
            new_size = max(width, height)
            final_clip = clip.on_color(size=(new_size, new_size), color=(0, 0, 0), col_opacity=1)
        else:  # 9:16 video
            new_size = max(width, height)
            final_clip = clip.on_color(size=(new_size, new_size), color=(0, 0, 0), col_opacity=1)

    output_path = f"output_with_padding_{target_aspect_ratio.replace(':', '_')}.mp4"
    return final_clip

# def convert_with_blur(clip, target_aspect_ratio):
#     """
#     Convert video to a new aspect ratio by adding a blurred background.

#     Parameters:
#     input_video (str): Path to the input video file.
#     target_aspect_ratio (str): Desired aspect ratio ("16:9", "9:16", "1:1").

#     Returns:
#     tuple: Path to the output video, new dimensions, and file size.
#     """
#     # clip = VideoFileClip(input_video)
#     width, height = clip.size
#     frame = clip.get_frame(0)
#     img = Image.fromarray(frame)

#     # Determine new size and blurred background
#     if target_aspect_ratio == "16:9":
#         blurred_img = img.resize((1280, 720)).filter(ImageFilter.GaussianBlur(20))
#         blurred_frame = np.array(blurred_img)
#         blurred_background = ImageClip(blurred_frame).set_duration(clip.duration)
#         main_clip = clip.fx(vfx.resize, height=720).set_position(("center", "center"))
#     elif target_aspect_ratio == "9:16":
#         blurred_img = img.resize((720, 1280)).filter(ImageFilter.GaussianBlur(20))
#         blurred_frame = np.array(blurred_img)
#         blurred_background = ImageClip(blurred_frame).set_duration(clip.duration)
#         main_clip = clip.fx(vfx.resize, width=720).set_position(("center", "center"))
#     elif target_aspect_ratio == "1:1":
#         new_size = max(width, height)
#         blurred_img = img.resize((new_size, new_size)).filter(ImageFilter.GaussianBlur(20))
#         blurred_frame = np.array(blurred_img)
#         blurred_background = ImageClip(blurred_frame).set_duration(clip.duration)
#         main_clip = clip.set_position(("center", "center"))

#     # Combine the blurred background and the resized main clip
#     final_clip = CompositeVideoClip([blurred_background, main_clip])

#     output_path = f"output_with_blur_{target_aspect_ratio.replace(':', '_')}.mp4"
#     # clip.close()
#     # blurred_background.close()
#     # main_clip.close()
#     # del img, blurred_img, blurred_frame, blurred_background, main_clip
#     return final_clip


def convert_with_blur(clip, target_aspect_ratio):
    import gc
    """
    Convert video to a new aspect ratio by adding a blurred background.

    Parameters:
    clip (VideoFileClip): Video clip to be processed.
    target_aspect_ratio (str): Desired aspect ratio ("16:9", "9:16", "1:1").

    Returns:
    CompositeVideoClip: Final clip with the new aspect ratio.
    """
    width, height = clip.size
    frame = clip.get_frame(0)
    img = Image.fromarray(frame)

    try:
        if target_aspect_ratio == "16:9":
            blurred_img = img.resize((1280, 720)).filter(ImageFilter.GaussianBlur(20))
            blurred_frame = np.array(blurred_img)
            blurred_background = ImageClip(blurred_frame).set_duration(clip.duration)
            main_clip = clip.fx(vfx.resize, height=720).set_position(("center", "center"))
        elif target_aspect_ratio == "9:16":
            blurred_img = img.resize((720, 1280)).filter(ImageFilter.GaussianBlur(20))
            blurred_frame = np.array(blurred_img)
            blurred_background = ImageClip(blurred_frame).set_duration(clip.duration)
            main_clip = clip.fx(vfx.resize, width=720).set_position(("center", "center"))
        elif target_aspect_ratio == "1:1":
            new_size = max(width, height)
            blurred_img = img.resize((new_size, new_size)).filter(ImageFilter.GaussianBlur(20))
            blurred_frame = np.array(blurred_img)
            blurred_background = ImageClip(blurred_frame).set_duration(clip.duration)
            main_clip = clip.set_position(("center", "center"))

        # Combine the blurred background and the resized main clip
        final_clip = CompositeVideoClip([blurred_background, main_clip])
    except Exception as e:
        print(e)
    finally:
        # Clean up resources
        if 'blurred_background' in locals():
            blurred_background.close()
        if 'main_clip' in locals():
            main_clip.close()

        # Delete objects and run garbage collector
        del img, blurred_img, blurred_frame
        gc.collect()

    return final_clip

from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, vfx
from PIL import Image, ImageFilter
import numpy as np

class VideoConverterWithBlur:
    def __init__(self, clip, target_aspect_ratio):
        """
        Initialize the VideoConverterWithBlur class.

        Parameters:
        clip (VideoFileClip): Loaded video clip object.
        target_aspect_ratio (str): Desired aspect ratio ("16:9", "9:16", "1:1").
        """
        self.clip = clip
        self.target_aspect_ratio = target_aspect_ratio
        self.blurred_background = None
        self.main_clip = None

    def convert(self):
        """
        Convert video to a new aspect ratio by adding a blurred background.

        Returns:
        final_clip (CompositeVideoClip): Final video clip with blurred background.
        """
        width, height = self.clip.size
        frame = self.clip.get_frame(0)
        img = Image.fromarray(frame)

        # Determine new size and blurred background
        if self.target_aspect_ratio == "16:9":
            blurred_img = img.resize((1280, 720)).filter(ImageFilter.GaussianBlur(20))
            blurred_frame = np.array(blurred_img)
            self.blurred_background = ImageClip(blurred_frame).set_duration(self.clip.duration)
            self.main_clip = self.clip.fx(vfx.resize, height=720).set_position(("center", "center"))
        elif self.target_aspect_ratio == "9:16":
            blurred_img = img.resize((720, 1280)).filter(ImageFilter.GaussianBlur(20))
            blurred_frame = np.array(blurred_img)
            self.blurred_background = ImageClip(blurred_frame).set_duration(self.clip.duration)
            self.main_clip = self.clip.fx(vfx.resize, width=720).set_position(("center", "center"))
        elif self.target_aspect_ratio == "1:1":
            new_size = max(width, height)
            blurred_img = img.resize((new_size, new_size)).filter(ImageFilter.GaussianBlur(20))
            blurred_frame = np.array(blurred_img)
            self.blurred_background = ImageClip(blurred_frame).set_duration(self.clip.duration)
            self.main_clip = self.clip.set_position(("center", "center"))

        # Combine the blurred background and the resized main clip
        final_clip = CompositeVideoClip([self.blurred_background, self.main_clip])

        return final_clip

    def close(self):
        """
        Close and cleanup any open intermediate objects.
        """
        if self.clip:
            self.clip.close()
        if self.blurred_background:
            self.blurred_background.close()
        if self.main_clip:
            self.main_clip.close()
        del self.clip, self.blurred_background, self.main_clip

# Usage example
# clip = VideoFileClip("input_video.mp4")
# converter = VideoConverterWithBlur(clip, "16:9")
# final_clip = converter.convert()
# final_clip.write_videofile("output_with_blur_16_9.mp4")
# converter.close()  # After final_clip has been used and closed elsewhere
