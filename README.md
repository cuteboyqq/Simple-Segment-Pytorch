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

below image Ref from paper Self-Attention Generative Adversarial Networks : https://arxiv.org/abs/1805.08318

![image](https://user-images.githubusercontent.com/58428559/230553064-bfed3fe9-0b0e-48c0-aa1f-77270a280328.png)

The main architecture network of CBAM is below :

The image is Ref from paper CBAM: Convolutional Block Attention Module : https://arxiv.org/abs/1807.06521

[(View Code)](Attention/CBAM/CBAM.py)

![image](https://user-images.githubusercontent.com/58428559/230556436-ec41313b-13b0-41cd-a329-2919944df7e8.png)

![image](https://user-images.githubusercontent.com/58428559/230556530-597caec6-40c7-413f-8d49-7de881f5949f.png)

## Unet
[(Back to top)](#table-of-contents)

[(View Code)](Unet_official/Unet.py)

below image Ref from paper U-Net: Convolutional Networks for Biomedical Image Segmentation : https://arxiv.org/abs/1505.04597

Ref code : https://github.com/milesial/Pytorch-UNet

![image](https://user-images.githubusercontent.com/58428559/230554890-98880bf8-104f-4b7f-a910-2132586f60b0.png)

## DeepLabV3
[(Back to top)](#table-of-contents)

[(View Code)](DeeplabV3/DeepLabV3_ver2.py)

Below Image is Ref. from Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation : https://arxiv.org/abs/1802.02611 

Ref. From Paper : Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

![image](https://user-images.githubusercontent.com/58428559/230640073-77cf2ab2-a070-41c3-9d9e-0872e1bbeb09.png)

![image](https://user-images.githubusercontent.com/58428559/230639982-d015cd80-2e6c-43bd-97fd-efe12b254ece.png)


## FCHardNet
[(Back to top)](#table-of-contents)
Base on Unet network, and use Resnet block to construct Unet

![image](https://user-images.githubusercontent.com/58428559/230553644-01db9dbd-62c7-461a-9b46-9308666f43db.png)


## Reference 
[(Back to top)](#table-of-contents)

1. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation :
https://arxiv.org/abs/1802.02611

2. U-Net: Convolutional Networks for Biomedical Image Segmentation :
https://arxiv.org/abs/1505.04597

3. CBAM: Convolutional Block Attention Module :
https://arxiv.org/abs/1807.06521

4. Self-Attention Generative Adversarial Networks :
https://arxiv.org/abs/1805.08318





