import torch
import torch.nn as nn

from functools import partial
from collections import OrderedDict

def drop_path(x, drop_prob:float = 0.,training:bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() 
    output = x.div(keep_prob) * random_tensor

    return output

class DropPath(nn.Module):
    def __init__(self,drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self,x):
        return drop_path(x, self.drop_prob, self.training)




class PatchEmbedding(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channel=3,
                 embed_dim=768,
                 norm_layer=None):
        super().__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0]//patch_size[0],
                          img_size[1]//patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channel, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None


    def forward(self,x):
        B,C,H,W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1],\
            f"input image size {H} is mismatched with expected size {self.img_size[0]} *{self.img_size[1]}"
        """
        这里设定了kernel size和stride都等于patch_size,
        也就是相当于卷积对原图像处理，处理的每一块都对应的一个Patch，
        768个卷积核即16x16x3，每个patch的大小为16x16，通道数为3，
        也就是一个patch的展平后的维度就是 768 

        对原图像进行这样的CNN处理后，每个patch相当于被处理了768次，得到的刚好就是其维度数
        而patch的数量为14x14=196
        """

        #batch size, channel, height, width
        #1,3,224,224 -> 1,768,14,14
        #flatten to 1,768,196
        #transpose to 1,196,768
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x

class Attention(nn.Module):
    def __init__(self,
                 dim=768,
                 num_heads=8,
                 qkv_bias=False,
                 qkv_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qkv_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.proj = nn.Linear(dim,dim)

    def forward(self,x):
        #batch size, num_patches+1, embedding_dim
        B,N,C = x.shape
        # tanspose to 3,B,num_heads,N,C//num_heads
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        # compute the dot product
        # shape of qkv:[B,num_heads,N,C//num_heads]
        # transpose(-2,-1) -> [B,num_heads,C//num_heads,N]
        #attn shape: [B,num_heads,N,N]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        # x shape: [B,num_heads,N,C//num_heads]
        # transpose :[B,N,num_heads,C//num_heads]
        x = (attn @ v).transpose(1,2).reshape(B,N,C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features = None, #通常是in_features * 4
                 out_features = None,
                 act_layer=nn.GELU,
                 drop_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Block(nn.Module):
    def __init__(self,
                 dim, #每个token的维度
                 num_heads,
                 mlp_ration = 4,
                 qkv_bias=False,
                 qkv_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads,
                              qkv_bias,
                              qkv_scale,
                              attn_drop_rate,
                              drop_rate)
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ration)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       out_features=dim,
                       act_layer=act_layer,
                       drop_rate=drop_rate)
        
    def forward(self,x):
        x = x+self.drop_path(self.attn(self.norm1(x)))
        x = x+self.drop_path(self.mlp(self.norm2(x)))

        return x
    

class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size = 224,
                 patch_size = 16,
                 in_channel = 3,
                 num_classes = 1000,
                 embed_dim = 768,
                 depth = 12,
                 num_heads = 12,
                 mlp_ratio = 4,
                 qkv_bias = True,
                 qkv_scale = None,
                 representation_size = None,
                 distilled = False,
                 drop_rate = 0.,
                 attn_drop_rate = 0.,
                 drop_path_rate = 0.,
                 embed_layer = PatchEmbedding,
                 norm_layer = None,
                 act_layer = None
                ):
        super().__init__()

        def _init_vit_weights(m):
            if isinstance(m, nn.Linear):
                # 对 Linear 层权重做标准正态分布初始化，标准差为 0.01
                nn.init.trunc_normal_(m.weight, std=.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Conv2d):
                # 对 Conv2d 层权重做 Kaiming 正态分布初始化，适合卷积层
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1

        norm_layer = norm_layer or partial(nn.LayerNorm,eps = 1e-6)
        act_layer = act_layer or nn.GELU()
        self.patch_embed = embed_layer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_channel=in_channel,
                                       embed_dim=embed_dim,
                                       norm_layer=norm_layer)
        num_patches = self.patch_embed.num_patches
        #class token shape: [batch size , token nums ,embed_dim]
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        if distilled:
            self.dist_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        else:
            self.dist_token = None
        
        #position embedding shape: [1,197,embed_dim]
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+self.num_tokens,embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.block = nn.Sequential(*[
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ration=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qkv_scale=qkv_scale,
                  drop_rate=drop_rate,
                  attn_drop_rate=attn_drop_rate,
                  drop_path_rate=dpr[i],
                  norm_layer=norm_layer,
                  act_layer=act_layer)
            for i in range(depth)])
        # the normalization after transformer
        self.norm = norm_layer(embed_dim)
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc",nn.Linear(embed_dim,representation_size)),
                        ("act",nn.Tanh())
                    ]
                )
            )
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        #classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.num_features, num_classes) if distilled and num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

        



    def forward_features(self,x):
        # x shape: [batch size, channel, height, width] -> [batch size, num_patches 196, embed_dim 768]
        x = self.patch_embed(x)
        #[1,1,768] -> [batch size, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  

        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token,self.dist_token.expand(x.shape[0],-1,-1),x),dim=1)
        
        x =self.pos_drop(x + self.pos_embed)
        x = self.block(x)
        x = self.norm(x)

        if self.dist_token is None:
            return self.pre_logits(x[:,0])
        else:
            return x[:,0], x[:,1]

    def forward(self,x):
        x = self.forward_features(x)
        if self.dist_token is not None:
            x , x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist

        else:
            x = self.head(x)
        return x


def vit_base_patch16_224(num_classes:int = 1000,
                         pretrained:bool = False,
                         pretrained_path:str = None,
                         **kwargs):

    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=None,
        num_classes=num_classes
    )

    if pretrained:
        if pretrained_path is None:
            raise ValueError("pretrained_path must be specified when pretrained is True")
        
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Load pretrained ViT model Successfully from {pretrained_path}")

    return model




 
        
