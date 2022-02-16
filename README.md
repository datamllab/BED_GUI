# BED: A Real-Time Object Detection System for Edge Devices
<img width="450" height="200" src="https://github.com/datamllab/BED_GUI/blob/main/figures/BED_logo.png">


### About this project

This project focuses on end-to-end oBject
detection system for Edge Devices (BED).
BED integrates a deep nerual network (DNN) practiced on [MAX78000](https://www.maximintegrated.com/en/products/microcontrollers/MAX78000.html) with I/O devices, as illustrated in the following figure. 
The DNN model for the detection is deployed on MAX78000; 
and the I/O devices include a camera and a screen for image acquisition and output exhibition, respectively. 

<div align=center>
<img width="450" height="200" src="https://github.com/datamllab/BED_GUI/blob/main/figures/GUI_pipeline.png">
</div>

### GUI ai8x Environment install

Before running GUI code, it is necessary to clone and install the software of [MAX78000 Evaluation Kit](https://github.com/MaximIntegratedAI/MaximAI_Documentation/tree/master/MAX78000_Evaluation_Kit).
Once finishing the installation, copy this repo to the root directory of "MAX78000_SDK/Examples/MAX78000/CNN/", and use this command to run the GUI:

````angular2html
conda env create -f ai8x.yml
conda activate ai8x
cd demo
python run_demo.py -c COM4
````

### GUI usuage

Click the **Load Image** button and then select the image in test_images folder.

<div align=center>
<img width="250" height="200" src="https://github.com/datamllab/BED_GUI/blob/main/figures/GUI_guide.png">
</div>

Once you selected the test image, it will show the detection results on the GUI, including the top-3 classification results and detection bounding box, as follows: 

<div align=center>
<img width="250" height="200" src="https://github.com/datamllab/BED_GUI/blob/main/figures/GUI_guide2.png">
</div>