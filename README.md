# AI

## Here are some of the projects I am most proud of 


### Soccer Object Detection

<img width="1144" alt="Screenshot 2024-03-30 at 11 17 44 AM" src="https://github.com/AngelicSage/AI/assets/142240060/387993c1-31c5-44c0-be22-2ccc515bec17">

You can watch the full video here:
https://drive.google.com/file/d/173caJcAJ7oT8RqtyH1YjvPpDtfU84fx1/view?usp=sharing

I used Roboflow to train this model, starting with the base Ms Coco model, and tuned it for my soccer dataset.


### Fine-tuned Stable Diffusion

Before:

![image](https://github.com/AngelicSage/AI/assets/142240060/8a7ae8e4-e4b5-4f9b-bcc1-58faaf367741)

After:

![Screenshot 2024-04-10 at 7 11 20 AM](https://github.com/AngelicSage/AI/assets/142240060/008d0d56-c44f-4db2-bbe7-973188319e84)

I used https://dreamlook.ai/dreambooth (a website running on AUTOMATIC1111) 

I simply had to use 50 images, and the improved so changed so much!

### GAN on celebrity faces

![dcgan](https://github.com/AngelicSage/AI/assets/142240060/e22f3787-891f-4fe1-ab2a-758f149b31ea)

Noise is decreased over each iteration to best resemble human features

I tried many different resolutions from 64x64 to 256x 256

50 epochs were used, more would be needed to capture all the features of humans.

### Image segmentation
![image](https://github.com/AngelicSage/AI/assets/142240060/9039480a-1aae-4a41-a7aa-aba03a0acdba)

Used pointrend due to their greater accuracy than mask r-cnn's

Coarse prediction -> subdivision steps(1-5)

 
### Fine tuned LLM for midjourney prompts

![Screenshot 2024-04-14 at 10 34 54 AM](https://github.com/AngelicSage/AI/assets/142240060/c7250db0-95a6-4acd-affa-d66df0064f4e)

I used the falcon 7b(a LLM from HuggingFace 

Fine tuned model with midjourney dataset


