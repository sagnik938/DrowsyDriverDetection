# DrowsyDriverDetection

Final Year project based on Deep Learning models and SVM to detect drowsiness  in heavy and light vehicle drivers. For more details read detailed project report.

<h2><li>Dependencies</h2>
<ul>python 3.8 or above</ul>
<ul>tensorflow</ul>
<ul>sklearn</ul>
<ul>opencv</ul>
<ul>matplotlib</ul>
<ul>numpy</ul>
</li>

<h2><li>Set up Environment</h2>
<ul>Install Python 3.8.9 <a href = "https://www.python.org/downloads/">Python</a></ul>
<ul>Install Tensorflow
    <ul><code>py -m pip install tensorflow</code>
    </ul>
</ul>
<ul>Install Numpy
    <ul><code>py -m pip install numpy</code>
    </ul>
</ul>
<ul>Install opencv
    <ul><code>py -m pip install opencv-python</code>
    </ul>
</ul>
<ul>Install sklearn
    <ul><code>py -m pip install scikit-learn</code>
    </ul>
</ul>
</li>
<h2><li>Model Descriptions<h2>
<h2><ul>RCNN Based Classifier :</h2>
        <p> This model implements a simple RCNN neural net with 0.25 dropouts, relu activation function for the CNN layer<br> followed by 2D max pooling and two dense layers 
             one having 128 (relu activated) and output layer having 4 neurons using softmax activation function.<br> The model uses adam optimizer and categorical cross-entropy as 
             the loss function.</p>
</ul>
<h2><ul>SVM Based Classifier :</h2>
        <p> This model implements a simple RCNN neural net with 0.25 dropouts, relu activation function for the CNN layer<br> followed by 2D max pooling and two dense layers 
             one having 128 (relu activated) and output layer having 4 neurons using l2 kernel regularization, activation function as softmax.<br>The Model compiles with adam optimizer and squared hinge as a loss function. A pre-trained sample is saved as SVM_1_20.h5.<br>Use the Keras load model to use and test and access the training report
             for detailed overview. </p>
</ul>
</li>



