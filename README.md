# Simple-Segment-Pytorch
[(Back to top)](#table-of-contents)

implement simple semantic segmentation network, you can regard these code as a outline of the network


### Table of contents

<!-- After you have introduced your project, it is a good idea to add a **Table of contents** or **TOC** as **cool** people say it. This would make it easier for people to navigate through your README and find exactly what they are looking for.

Here is a sample TOC(*wow! such cool!*) that is actually the TOC for this README. -->

- [Simple-Segment-Pytorch](#Simple-Segment-Pytorch)
- [Requirement](#Requirement)
- [implement](#implement)
- [Attention](#Attention)
- [Unet](#Unet)
- [FCHardNet](#FCHardNet)
- [DeepLabV3](#DeepLabV3)
- [Reference](#Reference)
   
### Requirement
[(Back to top)](#table-of-contents)
```
pip install -r requirements.txt
```

### implement 
[(Back to top)](#table-of-contents)

## Attention
[(Back to top)](#table-of-contents)

[(View Code)](Attention/Self-Attn-original/Self_Attention.py)

The main network of self-attention architecture is below :

below image Ref from paper : Self-Attention Generative Adversarial Networks  https://arxiv.org/abs/1805.08318

![image](https://user-images.githubusercontent.com/58428559/230553064-bfed3fe9-0b0e-48c0-aa1f-77270a280328.png)

The main architecture network of CBAM is below :

The image is Ref from paper : CBAM: Convolutional Block Attention Module https://arxiv.org/abs/1807.06521

[(View Code)](Attention/CBAM/CBAM.py)

![image](https://user-images.githubusercontent.com/58428559/230556436-ec41313b-13b0-41cd-a329-2919944df7e8.png)

![image](https://user-images.githubusercontent.com/58428559/230556530-597caec6-40c7-413f-8d49-7de881f5949f.png)

## Unet
[(Back to top)](#table-of-contents)

[(View Code)](Unet_official/Unet.py)

below image Ref from paper : U-Net: Convolutional Networks for Biomedical Image Segmentation https://arxiv.org/abs/1505.04597

Ref code : https://github.com/milesial/Pytorch-UNet

![image](https://user-images.githubusercontent.com/58428559/230554890-98880bf8-104f-4b7f-a910-2132586f60b0.png)

## DeepLabV3
[(Back to top)](#table-of-contents)

[(View Code)](DeeplabV3/DeepLabV3_ver2.py)

Below three Images is Ref. from paper : Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation https://arxiv.org/abs/1802.02611 

![image](https://user-images.githubusercontent.com/58428559/230640073-77cf2ab2-a070-41c3-9d9e-0872e1bbeb09.png)

![image](https://user-images.githubusercontent.com/58428559/230639982-d015cd80-2e6c-43bd-97fd-efe12b254ece.png)

![image](https://user-images.githubusercontent.com/58428559/230649662-c433f805-5ffc-4cd0-8aca-081eff0a6b8c.png)

### ASPP network study
below three images is Ref. from paper : Rethinking Atrous Convolution for Semantic Image Segmentation https://arxiv.org/abs/1706.05587

different method to get multi scale features

![image](https://user-images.githubusercontent.com/58428559/230705130-2ca8efe9-4535-46e1-b816-444391e886b3.png)

ASPP

![image](https://user-images.githubusercontent.com/58428559/230705017-ce0bb714-aed5-42ac-8da8-e527d936589f.png)

Atrous Convolution :

![image](https://user-images.githubusercontent.com/58428559/230705070-9a094c3b-7a00-435b-be18-bc1c23f7243c.png)


## FCHardNet
[(Back to top)](#table-of-contents)
Base on Unet network, and use Resnet block to construct Unet

![image](https://user-images.githubusercontent.com/58428559/230553644-01db9dbd-62c7-461a-9b46-9308666f43db.png)


## Reference 
[(Back to top)](#table-of-contents)

1. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation :
https://arxiv.org/abs/1802.02611

2. Rethinking Atrous Convolution for Semantic Image Segmentation 
https://arxiv.org/abs/1706.05587

3. U-Net: Convolutional Networks for Biomedical Image Segmentation :
https://arxiv.org/abs/1505.04597

4. CBAM: Convolutional Block Attention Module :
https://arxiv.org/abs/1807.06521

5. Self-Attention Generative Adversarial Networks :
https://arxiv.org/abs/1805.08318





