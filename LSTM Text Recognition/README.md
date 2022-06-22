# Text-Recognition-Using-RNN

### This Folder contains Text Recognition using Recurrent Neural Network with usage of tensorflow.

Dataset: [Tweet Emotion Dataset](https://github.com/dair-ai/emotion_dataset)

## Task 1:
It includes 2 plots firstly for confusion matrix and other one to evaluate the model, we also import necessary files here.

## Task 2:
It includes importing of Dataset from nlp.

Hugging Face, a company that first built a chat app for bored teens provides open-source NLP technologies, and last year, it raised $15 million to build a definitive NLP library. From its chat app to this day, Hugging Face has been able to swiftly develop language processing expertise. The company’s aim is to advance NLP and democratize it for use by everyone.

In a bid to make it easier for humans to communicate with machines, technologies such as NLP are crucial. For instance, with NLP, it is possible for computers to read text, hear speech, interpret it, measure sentiment, and even determine which parts of the text or speech are important. As more companies increasingly add NLP technologies for enhanced interactions, it becomes imperative to have ready libraries on which language models can be trained, saving time and cost. This is where companies like Hugging Face come into play. Its BERT models are considered highly effective and you can see them everywhere.

## Task 3:
Tokenizing the tweets

The tokenize module provides a lexical scanner for Python source code, implemented in Python. The scanner in this module returns comments as tokens as well, making it useful for implementing “pretty-printers”, including colorizers for on-screen displays.

To simplify token stream handling, all operator and delimiter tokens and Ellipsis are returned using the generic OP token type. The exact type can be determined by checking the exact_type property on the named tuple returned from tokenize.tokenize().

The tokenize() generator requires one argument, readline, which must be a callable object which provides the same interface as the io.IOBase.readline() method of file objects. Each call to the function should return one line of input as bytes.

The generator produces 5-tuples with these members: the token type; the token string; a 2-tuple (srow, scol) of ints specifying the row and column where the token begins in the source; a 2-tuple (erow, ecol) of ints specifying the row and column where the token ends in the source; and the line on which the token was found. The line passed (the last tuple item) is the physical line. The 5 tuple is returned as a named tuple with the field names: type string start end line.

## Task 4:
Padding and Truncating Sequences

Batched inputs are often different lengths, so they can’t be converted to fixed-size tensors. Padding and truncation are strategies for dealing with this problem, to create rectangular tensors from batches of varying lengths. Padding adds a special padding token to ensure shorter sequences will have the same length as either the longest sequence in a batch or the maximum length accepted by the model. Truncation works in the other direction by truncating long sequences.

In most cases, padding your batch to the length of the longest sequence and truncating to the maximum length a model can accept works pretty well. However, the API supports more strategies if you need them. The three arguments you need to are: padding, truncation and max_length.

The padding argument controls padding. It can be a boolean or a string:

True or 'longest': pad to the longest sequence in the batch (no padding is applied if you only provide a single sequence).
'max_length': pad to a length specified by the max_length argument or the maximum length accepted by the model if no max_length is provided (max_length=None). Padding will still be applied if you only provide a single sequence.
False or 'do_not_pad': no padding is applied. This is the default behavior.
The truncation argument controls truncation. It can be a boolean or a string:

True or 'longest_first': truncate to a maximum length specified by the max_length argument or the maximum length accepted by the model if no max_length is provided (max_length=None). This will truncate token by token, removing a token from the longest sequence in the pair until the proper length is reached.
'only_second': truncate to a maximum length specified by the max_length argument or the maximum length accepted by the model if no max_length is provided (max_length=None). This will only truncate the second sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided.
'only_first': truncate to a maximum length specified by the max_length argument or the maximum length accepted by the model if no max_length is provided (max_length=None). This will only truncate the first sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided.
False or 'do_not_truncate': no truncation is applied. This is the default behavior.

## Task 5:
Preparing the Labels, Creating classes to index and index to classes dictionaries and Converting text labels to numeric labels.

## Task 6:
Creation, training and evaluation of model.

## Task 7:
Creation of confusion matrix.

![image](https://user-images.githubusercontent.com/86974424/171842740-f9c79091-7c29-4d8a-bff3-45e79e5e53c6.png)
![image](https://user-images.githubusercontent.com/86974424/171842780-d50738e9-a0e1-4abe-b3e3-1e3b4074d6b7.png)
![image](https://user-images.githubusercontent.com/86974424/171842817-70e88aba-a881-40f7-b3d2-d6a1e30e6202.png)
![image](https://user-images.githubusercontent.com/86974424/171842847-33152db6-264c-4d3f-bf35-fa14cd2ff085.png)

## Padded Sequences

String padding is added to every string value so that one uniform order string can be passed through the model and evaluation could be easily seen on the histogram.

![image](https://user-images.githubusercontent.com/86974424/171843311-305bbf83-b7f9-4c4e-96d5-6021f9b25969.png)

## Labels Visualization

Here it can be seen that each and every tweets are divided into different classes according to the emotions they reflect: 

![image](https://user-images.githubusercontent.com/86974424/171843377-370215c1-7fbf-4ecd-8eae-5dfd07a4f394.png)

![image](https://user-images.githubusercontent.com/86974424/171843607-fb6bc0c0-a041-4104-85fd-ca948a1857ac.png)
![image](https://user-images.githubusercontent.com/86974424/171843631-219bd606-19c7-4e85-9612-82b35e2b6fa8.png)

## Model
![image](https://user-images.githubusercontent.com/86974424/171843507-17655e7a-2ab8-4758-b1e9-93cda10cb2e4.png)

![image](https://user-images.githubusercontent.com/86974424/171843683-67ff9b8c-17db-4127-95c8-fe31ae7a07a9.png)

## Model Evaluation
![image](https://user-images.githubusercontent.com/86974424/171843556-5bfcb43d-e917-4622-bfa0-3147b1ba5f0d.png)

![image](https://user-images.githubusercontent.com/86974424/171843715-5be217c4-d607-47ad-b0aa-79e47e78de02.png)
![image](https://user-images.githubusercontent.com/86974424/171843745-4295f943-bd81-41e5-b63c-9591ccbd9450.png)
#
# Web Application For Classification of Text/Comments/Tweets

As displayed in notebook, now through this application a user can test the model to do classfication of his or her text/comments/tweets into anger😠, joy😂, fear😨, sadness😞, surprise😯 or love💓. 

## Dashboard:

![image](https://user-images.githubusercontent.com/86974424/174992546-5671ee0f-60a0-43bc-8e66-b68d18ce7108.png)

## 1st Prediction

### Text: 

![image](https://user-images.githubusercontent.com/86974424/175002041-fa8854d8-441f-4c4a-9b46-65fdf44182de.png)

### Prediction:

![image](https://user-images.githubusercontent.com/86974424/175002158-a8c3dc99-1ef7-4739-b046-1ab939f0f780.png)

## 2nd Prediction:

### Text:

![image](https://user-images.githubusercontent.com/86974424/175002627-202636f5-8041-46f6-9b29-d80e6d6a570f.png)

### Prediction:

![image](https://user-images.githubusercontent.com/86974424/175002734-584bb0e5-6f2b-4d89-be67-3e747322723b.png)

##
## Text can be interpreted in many ways but according to the trained model we can see the first predciction is "surprise" and another is classified as "love" for the above inserted texts.
