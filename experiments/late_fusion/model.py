import torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class FusionModel(nn.Module):
    def __init__(self, image_model, text_model="albert", audio_model=None, video_model=None, vid_feats=2048, aud_feats=768,  img_feats=768, text_feats=768, mid_dim=1024, num_cls=1, dropout=0.1):
        super().__init__()
        self.audio_model = audio_model
        self.image_model = image_model
        self.text_model = text_model
        self.video_model = video_model
        self.img_feats = img_feats if image_model else 0
        self.text_feats = text_feats
        self.vid_feats = vid_feats if video_model else 0
        self.aud_feats = aud_feats if audio_model else 0
        self.mid_dim = mid_dim
        self.num_cls = num_cls
        self.dropout = dropout
        self.text_norm = nn.BatchNorm1d(text_feats)
        self.im_norm = nn.BatchNorm1d(img_feats)
        self.a_norm = nn.BatchNorm1d(aud_feats)
        self.v_norm = nn.BatchNorm1d(vid_feats)
        self.middle_layer = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.img_feats+self.aud_feats+self.vid_feats, self.mid_dim),
            nn.LayerNorm(self.mid_dim),
            nn.ReLU(inplace=True)
        )
        self.comb_layer = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.mid_dim + self.text_feats, self.mid_dim),
            nn.LayerNorm(self.mid_dim),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Sequential(
            nn.Linear(self.mid_dim*2, self.num_cls)
        )

    def forward(self, text_features, image1, video1, audio1, image2, video2, audio2):

        if self.image_model:
            im1_features = self.image_model(image1).pooler_output
            im1_features = self.im_norm(im1_features)

            im2_features = self.image_model(image2).pooler_output
            im2_features = self.im_norm(im2_features)
        else:
            im1_features = None
            im2_features = None

        a1_features = None
        a2_features = None
        if self.audio_model:
            a1_features = self.audio_model(audio1)
            a1_features = self.a_norm(a1_features)
            a2_features = self.audio_model(audio2)
            a2_features = self.a_norm(a2_features)
        
        v1_features = None
        v2_features = None
        if self.video_model:
            v1_features = video1.view(-1, 3*5, video1.size(2), video1.size(3))
            v1_features = v1_features.view(-1, 1, 8, 3*5, v1_features.size(2), v1_features.size(3))
            v1_features = self.video_model(v1_features)
            v1_features = self.v_norm(v1_features)

            v2_features = video2.view(-1, 3*5, video2.size(2), video2.size(3))
            v2_features = v2_features.view(-1, 1, 8, 3*5, v2_features.size(2), v2_features.size(3))
            v2_features = self.video_model(v2_features)
            v2_features = self.v_norm(v2_features)

        obj1_features = [im1_features, a1_features, v1_features]
        obj1_features = [i for i in obj1_features if i is not None]
        obj1_features = torch.cat(obj1_features, dim=1)
        obj1_features = self.middle_layer(obj1_features)

        obj2_features = [im2_features, a2_features, v2_features]
        obj2_features = [i for i in obj2_features if i is not None]
        obj2_features = torch.cat(obj2_features, dim=1)
        obj2_features = self.middle_layer(obj2_features)

        if self.text_model and self.text_feats > 0:
            text_features = self.text_norm(text_features)
            features1 = torch.cat([text_features, obj1_features], dim=1)
            features2 = torch.cat([text_features, obj2_features], dim=1)

            features1 = self.comb_layer(features1)
            features2 = self.comb_layer(features2)
        else:
            features1 = obj1_features
            features2 = obj2_features

        output = self.final_layer(torch.cat([features1, features2], dim=1))

        return output

