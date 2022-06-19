# Text-Classification-And-Recognition-With-TensorFlow

## Binary Classifier For World Bank Projects

### I used world bank project dataset from their official website with some pretrained BERT models to create the model. Using the project status feature I created a binary classifier model to predict whether a project will be sacked(Dropped) or completed(Closed) through its development objective.

Dataset: [World Bank Project Data](https://datacatalog.worldbank.org/search/dataset/0037800)

World Bank Projects & Operations provides access to basic information on all of the World Bank's lending projects from 1947 to the present. The dataset includes basic information such as the project title, task manager, country, project id, sector, themes, commitment amount, product line, and financing. It also provides links to publicly disclosed online documents.

1. Importing Necessary Packages:

Some of the Libraries such as Tensorflow, Scikit learn, Pandas, Numpy & Matplotlib are imported directly whereas Official Natural Language Processing Data and Bert Models are occupied throguh, [Official_NLP_Data_Bert_Model](https://github.com/tensorflow/models.git).

2. Dropping Null Values, Active and Pipeline status projects:

![image](https://user-images.githubusercontent.com/86974424/174475912-bf758c81-5a60-4704-9654-f0d6283bc8b4.png)

3. Converting project status feature to numeric data:

![image](https://user-images.githubusercontent.com/86974424/174475960-1f3fec58-46c1-4c50-915a-769ee8543e30.png)

4. Visualizing Item Count For Different Labels Of Dataset

![image](https://user-images.githubusercontent.com/86974424/174475970-9158b777-9be3-4e59-b0b5-f680b7d65626.png)

5. Tokenizing & Padding and Truncating Sequences:

![image](https://user-images.githubusercontent.com/86974424/174475993-fd3555f8-2c36-4bb8-8c2b-c7a60bbca095.png)

6. Model Summary:

![image](https://user-images.githubusercontent.com/86974424/174476013-27133c80-fff1-46ff-a952-a1555fa81f17.png)

7. Model View With Parameters:

![image](https://user-images.githubusercontent.com/86974424/174476024-81718f56-262c-4764-9311-dd8795b89a7f.png)

8. Model Evaluation

This is the loss and accuracy graph obtained from the model after training data is fitted here we can clearly see that after each epoch we can see decrease in loss from 45% to 20%, whereas validation loss do increase and due to which our accuracy also gets affected, seemingly accuracy rises upto 93% but validation accuracy decreases after 3rd epoch.

![image](https://user-images.githubusercontent.com/86974424/174476031-b140af6d-8aae-4c1e-bc27-79a29024ad99.png)

![image](https://user-images.githubusercontent.com/86974424/174476033-bb4f8432-3ab9-473d-bdcf-2ec5b4a96e66.png)

9. Confusion Matrix:

After Modelling, final plot is done with using confusion matrix and some data from the dataset throguh which our model successfully predicts 24 projects status from the 30 pojects and majorly predicts false positive values which means it predicts “closed” for the project status but in reality it is “dropped”.
Model Accuracy after testing – 88% 

![image](https://user-images.githubusercontent.com/86974424/174476046-6f7a6ce5-908b-46e7-876f-3274c1f3a2bb.png)

## Model Prediction:

Lastly You can test now taking any new project objective from world bank projects dataset and thus will be furnished with value of project status in [Sacked(Dropped) or Completed(Closed)]

![image](https://user-images.githubusercontent.com/86974424/174476212-2ffcfc02-6015-4476-adb2-b1f8ca7f52a0.png)

Here you can enter your project objective like I have entered "The program development objective is to support the Royal Government of Cambodia to effectively deal with the COVID-19 crisis and.................." and after running the cell you would be provided with the result like my sample project objective is predicted to be completed.

