# 🎾 3DCNN Tennis Action Recognition

This repository contains the full implementation of a project on tennis action recognition using **3D Convolutional Neural Networks**.  
The results and methods from this project are presented in the published paper:  
**“3D CNN Approach for Tennis Movement Recognition Using Spatiotemporal Features of Video”** in *Modelling and Simulation in Engineering*  
🔗[View the article](https://doi.org/10.1155/mse/1483523)

> 📌 This codebase was used in the research and directly reflects the methodology described in the article.

---

## 📝 Overview

The goal of this project is to automatically recognize tennis actions (e.g., serve, forehand, backhand) from video data using a 3D CNN architecture. The model works with either:
- raw **RGB frames**, or  
- **skeleton keypoint sequences**

This setup allows learning spatio-temporal representations to classify player actions effectively.

---

## 🚀 Getting Started

```bash
If you want to reproduce the training process, you need to:

1. clone this repo from link `https://github.com/Nelijahk/3DCNN-Tennis-Action-Recognition.git`.
2. run one of preprocessing `.py` based on what type of dataset you want to get (RGB or Skeleton).
3. now you can use one of training `.ipynb` to start training

   - > Please, make sure that the links are correctly written.
