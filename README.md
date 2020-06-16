# Speech Recognition and Lyric building

This project has been divided into two parts:
PART 1: Model building
PART 2: SPeech Recognition

## PART 1:
In this project, I have developed a recurrent neural network 
using Embeddings, Long Short Term Memory (LSTM) & Dense Layers.

The network was trained on the songs dataset from kaggle consisting of
of more than 60,000 songs from various artists. Each song had it's own 
associated lyric as well.

### Preprocessing data:
I performed most of the below mentioned tasks in google colab 
due to hardware restrictions on my personal system.

Firstly, I imported the dataset into google colab and filtered it
to consist of songs by 'Coldplay', 'Eminem' & 'Sia'. This was 
done to reduce the size of the dataset in order for the model to 
train faster.

I started by performing preliminary text preprocessing tasks
like :
1. Getting rid of the punctuations
3. Converting all text to lower case
2. Removing extra white spaces & empty lines if any

Note: We do not remove stopwords from the corpus, because we 
are not performing sentiment analysis; instead we are building 
a model to generate song lyrics, and stopwords are important 
for our use-case.

After performing these steps we form our corpus.
Then we tokenize our corpus by using the Tokenizer library 
of Tensorflow. We do this as in NLP it is critical to tokenize
your words for the model to better understand each words' importance.

Once, the words have been tokenized; I then used the concept of
N_gram_sequences and One_Hot_Encoding to generate Sequences & 
Labels.
	a. N-Gram Sequencing:
		It is the simplest model that assigns probabilities to 
		sentences and sequences of words.
		For in-depth understanding refer to ()
	b. One_Hot_Encoding:
		In this, categorical varaiables are provided into a form
		that will help algorithms to do a better job in prediction.
		For in-depth understandig refer to ()

We have used 'pad_sequences' library as well provided by
Tensorflow to make sure that each input in the form of n-gram sequence
for the network is of the same length. After doing so, we split
the sequence such that all the tokens except the last token are
treated as input and the last token is treated as 'label'. We
then one-hot encode the labels for better prediction purposes.

Now that our feature i.e. input and label i.e. one-hot encoded label
are ready, we can start building a neural network.

### Generating Model:
Building an RNN to train our text generation model using 
categorical_crossentorpy as our loss and accuracy as our 
metric.

We will be using an Embedding Layer to create n-embedding dimension 
to enhance our performance. We will the ouput of the embedding layer to 
the bi-directional Long Short Term Memory (LSTM) layer. It's remember 
gate and forget gate will play a big part in predicting the lyric. 
We will add or remove these depending on the system performace. 
Finally, we have a Dense layer as our ouptut layer with the activation 
function 'softmax'.

We built 4 different models by changing the value embedding dimension, adding
kernel (filter), adding more LSTM levels, increasing epoch levels etc.

In each model we observe that accuracy starts at a very low value and 
increases until it reaches a plateau. Each model on an average takes more 
than 30 mins to run. So, it is very hard to attain an accuracy of 90% with
memory & RAM restrictions. Our best model achieved an accuracy of 83%; but that
network is very exhausting to the system.

We save our best model as a SavedModel using Tensorflow, so that we can 
import it in another file and do not have to re-run it all over again.
By doing this we can save a ton of time. Saving the model saves the biases 
and weights attached to the hidden layers. This means that the model won't 
be compromised when working in a different environment.

After saving it, we download the model to our loacal machine.

## PART 2:
The user will provide speech input in this case, song lyric
to the system using PyAudio and SpeechRecognition libraries of Python 
for speech detection, and our model will predict the next words of the 
song.
After doing so, the system will complete the song lyric and output it 
in audio format using the Google text to Speech (gTTS) library.
 
This is the same model we built in PART 1 with 83% accuracy.
We load this model in our local system using Tensorflow's load_model
functionality.
 
I have implemented this part in my local system to demonstrate how models 
can be saved and loaded in different environments irrespective of where it 
has been built without compromising on it's performance. Also, as there 
were issues when google colab had to acces my microphone for audio input.
 
We import the dataset, retreive the corpus and word_tokens as we did
in PART 1. We do this as the model predicts the next word in a sequence based
on it's token value. Hence, we restrict the dataset with songs by Coldplay,
Eminem, & Sia.
 
The users audio input is converted into text using speechRecognition.
This text is then converted to sequences sing Tokenizer.
We pad it so that the input size is similar to that on which the model has 
been trained.
These steps are similar to what we accomplished in PART 1.
The model predicts the next words based on the tokens and ives the output 
in mp3 format. We then use os library to play this mp3.
 
eg. User input- sky full of stars
Model Predicts- i want to die in your arms
system returns- sky full of stars i want to die in your arms
 
Our model was successfully able to predict the next words of the song
'A Sky Full of Stars' by Coldplay. Incase the model enctoures a situation 
wherein the next words of a lyric, it will output words that it deems to 
see fit to the associated input.

The link to the video:
https://drive.google.com/file/d/1QOzKytLBrOgvN08IUCjisLThIGLn93aC/view?usp=sharing
