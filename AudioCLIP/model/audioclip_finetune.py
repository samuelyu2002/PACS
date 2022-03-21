import os

import torch
import torch.nn.functional as F

from model.clip import CLIP
from model.clip.clip import tokenize
# from clip import tokenize
from model.esresnet import ESResNeXtFBSP

from typing import List
from typing import Tuple
from typing import Union
from typing import Optional


ClipFeatures = Tuple[
    Optional[torch.Tensor],  # audio
    Optional[torch.Tensor],  # image
    Optional[torch.Tensor]   # audio
]


ClipLogits = Tuple[
    Optional[torch.Tensor],  # audio x image
    Optional[torch.Tensor],  # audio x text
    Optional[torch.Tensor]   # image x text
]


ClipOutput = Tuple[
    Tuple[ClipFeatures, ClipLogits],
    Optional[torch.Tensor]   # loss
]


class AudioCLIPFinetune(CLIP):

    def __init__(self,
                 embed_dim: int = 1024,
                 # vision
                 image_resolution: int = 224,
                 vision_layers: Union[Tuple[int, int, int, int], int] = (3, 4, 6, 3),
                 vision_width: int = 64,
                 vision_patch_size: Optional[int] = None,
                 # text
                 context_length: int = 77,
                 vocab_size: int = 49408,
                 transformer_width: int = 512,
                 transformer_heads: int = 8,
                 transformer_layers: int = 12,
                 # audio
                 n_fft: int = 2048,
                 hop_length: Optional[int] = 561,
                 win_length: Optional[int] = 1654,
                 window: Optional[str] = 'blackmanharris',
                 normalized: bool = True,
                 onesided: bool = True,
                 spec_height: int = -1,
                 spec_width: int = -1,
                 apply_attention: bool = True,
                 multilabel: bool = True,
                 pretrained: Union[bool, str] = True):

        super(AudioCLIPFinetune, self).__init__(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers
        )

        self.audio = ESResNeXtFBSP(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=embed_dim,
            apply_attention=apply_attention,
            pretrained=False
        )

        self.multilabel = multilabel
        self.pretrained = pretrained

        self.logit_scale_ai = torch.nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_at = torch.nn.Parameter(torch.log(torch.ones([]) * 100))

        if isinstance(self.pretrained, str):
            self.load_state_dict(torch.load(self.pretrained, map_location='cpu'), strict=False)
        elif self.pretrained:
            self.load_state_dict(torch.load(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'CLIP.pt'),
                map_location='cpu'
            ), strict=False)
            print('Image & Text weights loaded')
            try:
                self.audio.load_state_dict(torch.load(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'ESRNXFBSP.pt'),
                    map_location='cpu'
                ), strict=False)
            except RuntimeError as ex:
                print(ex)
                print('Audio weights loaded')

        self.embed_dim = embed_dim

        self.audio_image_fuse = torch.nn.Linear(embed_dim*2, 512)

    @property
    def device(self):
        return self.visual.conv1.weight.device

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return self.audio(audio.to(self.device))

    def encode_text(self, tokens):
        return super(AudioCLIPFinetune, self).encode_text(tokens)

    def forward(self, image1, audio1, image2, audio2, tokens):

        audio1 = audio1.squeeze()
        audio2 = audio2.squeeze()
        
        audio_features1 = self.encode_audio(audio1)
        audio_features1 = audio_features1 / audio_features1.norm(dim=-1, keepdim=True)
        image_features1 = self.encode_image(image1)
        image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)

        audio_features2 = self.encode_audio(audio2)
        audio_features2 = audio_features2 / audio_features2.norm(dim=-1, keepdim=True)
        image_features2 = self.encode_image(image2)
        image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)

        obj1_features = self.audio_image_fuse(torch.cat([audio_features1, image_features1], dim=1))
        obj2_features = self.audio_image_fuse(torch.cat([audio_features2, image_features2], dim=1))

        obj1_features = obj1_features / obj1_features.norm(dim=-1, keepdim=True)
        obj2_features = obj2_features / obj2_features.norm(dim=-1, keepdim=True)

        text_features = self.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # print(text_features.shape)
        # print(obj1_features.shape)
        # assert(1==0)

        return obj1_features, obj2_features, text_features
        # return audio_features1, audio_features2