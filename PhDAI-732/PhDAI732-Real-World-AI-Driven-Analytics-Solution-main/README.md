# AI Recommendation System (Machine-Learning-on-Movie-Reviews-IMDB)
## Objective  

The goal of this project is to design, implement, and evaluate a Python-based data analytics solution that addresses a real-world business or societal problem using core Python libraries and machine learning tools. Students will apply data preprocessing, exploratory data analysis, model development, and ethical evaluation, using techniques such as regression, classification, or clustering to deliver AI-driven insights. 
 Students will work in teams to simulate the lifecycle of a data analytics project, including stakeholder problem framing, iterative development, model evaluation, and reflection on ethical use of AI. 

### Dataset: 
You will typically use a well-known dataset, such as the MovieLens 100k dataset, which contains information about user ratings for different movies. The dataset includes: 
- user_id: Unique identifier for users. 
- item_id: Unique identifier for items (e.g., movies). 
- rating: Rating given by the user to the item (usually between 1 and 5). 

### Steps for the Project: 
#### 1. Understanding Recommendation Systems 

Focus on collaborative filtering: This method uses user-item interactions to recommend items to users based on similarities with other users (user-based) or items (item-based). 
Understand the difference between collaborative filtering and content-based filtering. Collaborative filtering relies on the assumption that if users agree in the past, they will agree in the future on items. 
### 2. Data Exploration and Preprocessing 

Load and inspect the dataset. 
Data cleaning: Handle missing values, duplicates, and ensure correct data types (e.g., ensure user_id and item_id are integers, and ratings are floats). 
Visualize data to gain insights into the distribution of ratings, number of ratings per user, etc. 
### 3. Data Splitting 

Split the dataset into training and testing sets. The training set will be used to train the recommendation model, and the testing set will help evaluate its performance. 
Use train_test_split from the Surprise library to make this process easier. 
from surprise.model_selection import train_test_split 

trainset, testset = train_test_split(data_surprise, test_size=0.25) 

### 4. Model Selection 

SVD (Singular Value Decomposition) is a popular algorithm for collaborative filtering. It works well for recommendation tasks. 
Use the Surprise library, which provides implementations of several recommendation algorithms, including SVD. 
from surprise import SVD 

model = SVD() 

model.fit(trainset) 

### 5. Model Evaluation 

Evaluate the model using Root Mean Squared Error (RMSE), a common metric for measuring the accuracy of recommendation models. 
After making predictions, calculate RMSE: 
from surprise import accuracy 

rmse = accuracy.rmse(predictions) 

print(f"Root Mean Squared Error: {rmse}") 

### 6. Hyperparameter Tuning 

Hyperparameters such as n_factors (number of latent factors), n_epochs (number of iterations), and lr_all (learning rate) can be tuned to improve model performance. 
Use GridSearchCV to find the best hyperparameters for the SVD model. 
from surprise.model_selection import GridSearchCV 

param_grid = {'n_factors': [50, 100], 'n_epochs': [20, 40], 'lr_all': [0.002, 0.005]} 

grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3) 

grid_search.fit(data_surprise) 

print(grid_search.best_params) 

### 7. Top N Recommendations 

Once the model is trained, generate the top N recommendations for each user. 
Sort predicted ratings and recommend the highest-rated items for each user. 
def get_top_n(predictions, n=10): 

   top_n = {} 

   for uid, iid, true_r, est, _ in predictions: 

       if uid not in top_n: 

           top_n[uid] = [] 

       top_n[uid].append((iid, est)) 

 

   for uid, user_ratings in top_n.items(): 

       user_ratings.sort(key=lambda x: x[1], reverse=True) 

       top_n[uid] = user_ratings[:n] 

 

   return top_n 

### 8. Ethical Considerations 

Reflect on ethical issues in recommendation systems, such as: 
Bias: Ensure that the data used for training is not biased toward certain demographic groups. 
Privacy: Be mindful of user data and ensure it is handled properly. 
Transparency: Ensure users understand how recommendations are made. 

## Tools and Libraries: 
- Python: Programming language used for the project. 
- Surprise Library: A Python library for building recommendation systems. 
- pandas: For data manipulation. 
- matplotlib/seaborn: For data visualization. 
- Google Colab: For running the code and sharing the project. 

This code will help us understand how to perform sentiment analysis and machine learning on a movie review dataset.

This repository uses the IMDB movie reviews dataset containing 50k movie reviews to create a sentiment analysis and machine learning model. The dataset is highly balanced and consists of 25,000 highly polar positive and negative movie reviews. The various machine learning classifiers used in this project include Logistic Regression, Multinomial Naive Bayes, and Linear Support Vector Classifiers.

## Original Data Source - https://files.grouplens.org/datasets/movielens/ml-100k.zip
