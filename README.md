# Emotion-Detection-in-Speech

The information in speech is conveyed through words and emotion. Depending on how one 
pronounces a word, we can understand different meaning. Therefore, both the word 
and the way it is pronounced affect our understaning of the word. As a result, 
it would be important that we could design a virtual assistant that not onldy 
does understand the word, but also it understands the emotion in the way the word
is pronounced. 

In this work, we are trying to introduce a machine learning model that can detect the
emotion in the sentence. In order to do so, we are using the audio data provided 
by [University of Toronto](https://tspace.library.utoronto.ca/handle/1807/24487) 
to create a neural network model to detect the emotion in a speech. In oreder to 
make a model we need to convert the audio files into numerical values. To convert
the audio file to a numerical data, we use python library [librosa](https://librosa.org/)
and to denoise them we used [noisereduce](https://pypi.org/project/noisereduce/).
Afterward we convert the denoised audio files into their corresponding spectrograms 
using which we can train a Convolutional neural network. 


The arrangment of this work is as follows:

1. The folder `Toronto-Data` contains the audio files we used in this work.
2. The folder `Notebook` contains the following notebooks:
    * The Exploratory Data Analysis (EDA), Train-Test split and conversion of
    audio files to spectrograms are done in the notebook `EDA-and-Data-Visualization.ipynb`.
3. The folder `Train-Test-Split` contains the training and testing dataframes.
4. The folder `mel_spectrogram` contains the spectrograms of the audio files.




