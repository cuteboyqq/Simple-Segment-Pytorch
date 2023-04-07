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
   
### Requirement
[(Back to top)](#table-of-contents)
```
pip install -r requirements.txt
```

### implement 
[(Back to top)](#table-of-contents)

### Attention
[(Back to top)](#table-of-contents)

The main network of self-attention architecture is below :
Ref from paper https://arxiv.org/abs/1805.08318
![image](https://user-images.githubusercontent.com/58428559/230553064-bfed3fe9-0b0e-48c0-aa1f-77270a280328.png)

### Unet
[(Back to top)](#table-of-contents)

Ref code : https://github.com/milesial/Pytorch-UNet


### FCHardNet
[(Back to top)](#table-of-contents)
Base on Unet network, and use Resnet block to construct Unet

![image](https://user-images.githubusercontent.com/58428559/230553644-01db9dbd-62c7-461a-9b46-9308666f43db.png)


### Reference 
[(Back to top)](#table-of-contents)


