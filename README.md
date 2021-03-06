# wind-scale-estimation-with-dual-branch-network
This code repository is for our paper "See the Wind: Wind Scale Estimation with Optical Flow and VisualWind
Dataset"

Authors: Qin Zhang, Jialang Xu, Matthew Crane, Chunbo Luo

The purpose is to estimate the wind scale from videos. To address this problem, we build a visual dataset named VisualWind which is the first of its kind video dataset, collecting videos with trees swaying under various scales of wind from social media, public cameras and self-recording. The link for the dataset is as blow. 
https://sme.uds.exeter.ac.uk/folders/48caf5102d6196b9645fab1f46e494ec. An access key is needed to get to the dataset due to the server security policy. Please contact the author to get the access key if you are interested. 

We propose a dual-branch deep learning model to estimate the wind scales in an end-to-end manner, consisting of a motion branch to extract motion features by optical flow, and a visual branch to extract visual features by convolutional operation, and achieving 86.69\% accuracy on the proposed dataset.

![Graphical abstract](https://github.com/qinzhang2016/wind-scale-estimation-with-dual-branch-network/blob/main/figures/graphicalabstract.png)


