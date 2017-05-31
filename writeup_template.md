# **Finding Lane Lines on the Road by Manuel Quinteiro** 


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Pipeline.

I've modified the proposed pipeline to:

1) Cut the top 1/2 of the image to speedup calculus
2) erase all the dots that not have a minimun values, with a lower limit for blue to avoid yellow lines problems.
3) apply Canny algorithm 
4) Apply hought_lines to canny output.

I dind't use ROI because I want to check if I can detect just my line between some other lanes.

Regarding the draw_lines I've chage a lot.
Fisrt I take only the lines that cross one reference, I want only lines that cros a reference horizontal line at 4/5 of the screen, also I take only the lines with hi slope, and finally I discart hi chagnes in the slope and position (this is just for chalenge video)

For the challenge.mp4 video I need more complexity to the system so I give a kind of memory that I have call Histeresys 
absorving up tu N errors in the slope and position of the lines.

If you'd like to include images of the videos and static ones to show how the pipeline works, 

[image2]: ./test_images_output/video2.png


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when lot of light in the image, I did some kind of tricks to avoid but it can happend

Also I can have problems in the video with slow movement.




### 3. Suggest possible improvements to your pipeline

adjustment of the image light before using canny algorithm could be good.
avoid all kind of restriction to the image also could be better solution.