# Third-party
import torch

# Global Variables
COMPUTATION_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXECUTION_PROVIDER_LIST = ["CUDAExecutionProvider", "CPUExecutionProvider"]
ONNX_MODEL_PATH = "Kim_Vocal.onnx"

# import torch
import onnxruntime as ort

# Local Imports
# from src.models.MDX_net.mdx_net import Conv_TDF_net_trimm
# from src.loader import Loader

# Global Variables
# from src.constants import EXECUTION_PROVIDER_LIST, COMPUTATION_DEVICE, ONNX_MODEL_PATH

import os

# Explicit Typing
from typing import Tuple
from numpy import ndarray

# Third-party
# import librosa


# class Loader:
#     """Loading sound files into a usable format for pytorch"""

#     def __init__(self, INPUT_FOLDER, OUTPUT_FOLDER):
#         self.input = INPUT_FOLDER
#         self.output = OUTPUT_FOLDER
#     def load_wav(self, name) -> Tuple[ndarray, int]:
#         music_array, samplerate = librosa.load(
#             os.path.join(self.input, name + ".wav"), mono=False, sr=44100
#         )
#         return music_array, samplerate

#     def prepare_uploaded_file(self, uploaded_file) -> Tuple[torch.Tensor, int]:
#         music_array, samplerate = librosa.load(uploaded_file, mono=False, sr=44100)

#         music_tensor = torch.tensor(music_array, dtype=torch.float32)

#         return music_tensor, samplerate


import torch.nn as nn



class STFT:
    def __init__(self, n_fft, hop_length, dim_f):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=n_fft, periodic=True)
        self.dim_f = dim_f

    def __call__(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([*batch_dims, c, 2, -1, x.shape[-1]]).reshape(
            [*batch_dims, c * 2, -1, x.shape[-1]]
        )
        return x[..., : self.dim_f, :]

    def inverse(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]
        n = self.n_fft // 2 + 1
        f_pad = torch.zeros([*batch_dims, c, n - f, t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        t_complex = torch.view_as_complex(x)
        x = torch.istft(
            t_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
        )
        x = x.reshape([*batch_dims, 2, -1])
        return x

class Conv_TDF(nn.Module):
    """
    Convolutional Time-Domain Filter (TDF) Module.

    Args:
        c (int): The number of input and output channels for the convolutional layers.
        l (int): The number of convolutional layers within the module.
        f (int): The number of features (or units) in the time-domain filter.
        k (int): The size of the convolutional kernels (filters).
        bn (int or None): Batch normalization factor (controls TDF behavior). If None, TDF is not used.
        bias (bool): A boolean flag indicating whether bias terms are included in the linear layers.

    Attributes:
        use_tdf (bool): Flag indicating whether TDF is used.

    Methods:
        forward(x): Forward pass through the TDF module.
    """

    def __init__(self, c, l, f, k, bn, bias=True):
        super(Conv_TDF, self).__init__()

        # Determine whether to use TDF (Time-Domain Filter)
        self.use_tdf = bn is not None

        # Define a list of convolutional layers within the module
        self.H = nn.ModuleList()
        for i in range(l):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=c,
                        out_channels=c,
                        kernel_size=k,
                        stride=1,
                        padding=k // 2,
                    ),
                    nn.GroupNorm(2, c),
                    nn.ReLU(),
                )
            )

        # Define the Time-Domain Filter (TDF) layers if enabled
        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias), nn.GroupNorm(2, c), nn.ReLU()
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    nn.GroupNorm(2, c),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    nn.GroupNorm(2, c),
                    nn.ReLU(),
                )

    def forward(self, x):
        # Apply the convolutional layers sequentially
        for h in self.H:
            x = h(x)

        # Apply the Time-Domain Filter (TDF) if enabled, and add the result to the orignal input
        return x + self.tdf(x) if self.use_tdf else x


class Conv_TDF_net_trimm(nn.Module):
    """
    Convolutional Time-Domain Filter (TDF) Network with Trimming.

    Args:
        L (int): This parameter controls the number of down-sampling (DS) blocks in the network.
                 It's divided by 2 to determine how many DS blocks should be created.
        l (int): This parameter represents the number of convolutional layers (or filters) within each dense (fully connected) block.
        g (int): This parameter specifies the number of output channels for the first convolutional layer and is also used to determine the number of channels for subsequent layers in the network.
        dim_f (int): This parameter represents the number of frequency bins (spectrogram columns) in the input audio data.
        dim_t (int): This parameter represents the number of time frames (spectrogram rows) in the input audio data.
        k (int): This parameter specifies the size of convolutional kernels (filters) used in the network's convolutional layers.
        bn (int or None): This parameter controls whether batch normalization is used in the network.
                         If it's None, batch normalization may or may not be used based on other conditions in the code.
        bias (bool): This parameter is a boolean flag that controls whether bias terms are included in the convolutional layers.
        overlap (int): This parameter specifies the amount of overlap between consecutive chunks of audio data during processing.

    Attributes:
        n (int): The calculated number of down-sampling (DS) blocks.
        dim_f (int): The number of frequency bins (spectrogram columns) in the input audio data.
        dim_t (int): The number of time frames (spectrogram rows) in the input audio data.
        n_fft (int): The size of the Fast Fourier Transform (FFT) window.
        hop (int): The hop size used in the STFT calculations.
        n_bins (int): The number of bins in the frequency domain.
        chunk_size (int): The size of each chunk of audio data.
        target_name (str): The name of the target instrument being separated.
        overlap (int): The amount of overlap between consecutive chunks of audio data during processing.

    Methods:
        forward(x): Forward pass through the Conv_TDF_net_trimm network.
    """

    def __init__(
        self,
        model_path,
        use_onnx,
        target_name,
        L,
        l,
        g,
        dim_f,
        dim_t,
        k=3,
        hop=1024,
        bn=None,
        bias=True,
        overlap=1500,
    ):
        super(Conv_TDF_net_trimm, self).__init__()
        # Dictionary specifying the scale for the number of FFT bins for different target names
        n_fft_scale = {"vocals": 3, "*": 2}

        # Number of input and output channels for the initial and final convolutional layers
        out_c = in_c = 4

        # Number of down-sampling (DS) blocks
        self.n = L // 2

        # Dimensions of the frequency and time axes of the input data
        self.dim_f = 3072
        self.dim_t = 256

        # Number of FFT bins (frequencies) and hop size for the Short-Time Fourier Transform (STFT)
        self.n_fft = 7680
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1

        # Chunk size used for processing
        self.chunk_size = hop * (self.dim_t - 1)

        # Target name for the model
        self.target_name = target_name

        # Overlap between consecutive chunks of audio data during processing
        self.overlap = overlap

        # STFT module for audio processing
        self.stft = STFT(self.n_fft, self.hop, self.dim_f)

        # Check if ONNX representation of the model should be used
        if not use_onnx:
            # First convolutional layer
            self.first_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=g, kernel_size=1, stride=1),
                nn.BatchNorm2d(g),
                nn.ReLU(),
            )

            # Initialize variables for dense (fully connected) blocks and downsampling (DS) blocks
            f = self.dim_f
            c = g
            self.ds_dense = nn.ModuleList()
            self.ds = nn.ModuleList()

            # Loop through down-sampling (DS) blocks
            for i in range(self.n):
                # Create dense (fully connected) block for down-sampling
                self.ds_dense.append(Conv_TDF(c, l, f, k, bn, bias=bias))

                # Create down-sampling (DS) block
                scale = (2, 2)
                self.ds.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=c,
                            out_channels=c + g,
                            kernel_size=scale,
                            stride=scale,
                        ),
                        nn.BatchNorm2d(c + g),
                        nn.ReLU(),
                    )
                )
                f = f // 2
                c += g

            # Middle dense (fully connected block)
            self.mid_dense = Conv_TDF(c, l, f, k, bn, bias=bias)

            # If batch normalization is not specified and mid_tdf is True, use Conv_TDF with bn=0 and bias=False
            if bn is None and mid_tdf:
                self.mid_dense = Conv_TDF(c, l, f, k, bn=0, bias=False)

            # Initialize variables for up-sampling (US) blocks
            self.us_dense = nn.ModuleList()
            self.us = nn.ModuleList()

            # Loop through up-sampling (US) blocks
            for i in range(self.n):
                scale = (2, 2)
                # Create up-sampling (US) block
                self.us.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels=c,
                            out_channels=c - g,
                            kernel_size=scale,
                            stride=scale,
                        ),
                        nn.BatchNorm2d(c - g),
                        nn.ReLU(),
                    )
                )
                f = f * 2
                c -= g

                # Create dense (fully connected) block for up-sampling
                self.us_dense.append(Conv_TDF(c, l, f, k, bn, bias=bias))

            # Final convolutional layer
            self.final_conv = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=out_c, kernel_size=1, stride=1),
            )

            try:
                # Load model state from a file
                self.load_state_dict(
                    torch.load(
                        f"{model_path}/{target_name}.pt",
                        map_location=COMPUTATION_DEVICE,
                    )
                )
                print(f"Loading model ({target_name})")
            except FileNotFoundError:
                print(f"Random init ({target_name})")

    def forward(self, x):
        """
        Forward pass through the Conv_TDF_net_trimm network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.first_conv(x)

        x = x.transpose(-1, -2)

        ds_outputs = []
        for i in range(self.n):
            x = self.ds_dense[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        x = self.mid_dense(x)

        for i in range(self.n):
            x = self.us[i](x)
            x *= ds_outputs[-i - 1]
            x = self.us_dense[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        return x



class KimVocal:
    """
    TODO: Put something here for flexibility purposes (model types).
    """

    def __init__(self):
        pass

    def demix_vocals(self, music_tensor, sample_rate, model):
        """
        Removing vocals using a ONNX model.

        Args:
            music_tensor (torch.Tensor): Input tensor.
            model (torch.nn): Model used for inferring.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        number_of_samples = music_tensor.shape[1]
        overlap = model.overlap
        # Calculate chunk_size and gen_size based on the sample rate
        chunk_size = model.chunk_size
        gen_size = chunk_size - 2 * overlap
        pad_size = gen_size - number_of_samples % gen_size
        mix_padded = torch.cat(
            [torch.zeros(2, overlap), music_tensor, torch.zeros(2, pad_size + overlap)],
            1,
        )

        # Start running the session for the model
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=EXECUTION_PROVIDER_LIST)

        # TODO: any way to optimize against silence? I think that's what skips are for, gotta double check.
        # process one chunk at a time (batch_size=1)
        demixed_chunks = []
        i = 0
        while i < number_of_samples + pad_size:
            # Progress Bar

            # Computation
            chunk = mix_padded[:, i : i + chunk_size]
            x = model.stft(chunk.unsqueeze(0).to(COMPUTATION_DEVICE))
            with torch.no_grad():
                x = torch.tensor(ort_session.run(None, {"input": x.cpu().numpy()})[0])
            x = model.stft.inverse(x).squeeze(0)
            x = x[..., overlap:-overlap]
            demixed_chunks.append(x)
            i += gen_size

        vocals_output = torch.cat(demixed_chunks, -1)[..., :-pad_size].cpu()

        return vocals_output

# loader = Loader(INPUT_FOLDER, OUTPUT_FOLDER)
# music_tensor, samplerate = loader.prepare_uploaded_file(
#     uploaded_file=
# )

# music_array, samplerate = librosa.load("test.mp3", mono=False, sr=44100)

# music_tensor = torch.tensor(music_array, dtype=torch.float32)

# model_raw_python = Conv_TDF_net_trimm(
#     model_path=ONNX_MODEL_PATH,
#     use_onnx=True,
#     target_name="vocals",
#     L=11,
#     l=3,
#     g=48,
#     bn=8,
#     bias=False,
#     dim_f=11,
#     dim_t=8,
# )

# kimvocal = KimVocal()
# vocals_tensor = kimvocal.demix_vocals(
#     music_tensor=music_tensor,
#     sample_rate=samplerate,
#     model=model_raw_python,
# )
# vocals_array = vocals_tensor.numpy()
# print(vocals_array)
# # Update progress

