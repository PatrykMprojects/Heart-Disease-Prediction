# Heart-disease-prediction
Due to the size of the Heart_Disease_Prediction_comparison file, it might not show in GitHub therefore it is advised to download the file and load it into Collab or any other Python file reader.  

This project comprehensively evaluates machine learning algorithms for classifying whether a patient has heart disease. Additionally, I deploy the best-performing model into a Streamlit web application, allowing for seamless interaction and accessibility.

The main objective of this project is to combine the TabPFN transformer with machine learning models using ensemble methods for heart disease prediction. The performance of 14 algorithms were tested on the three different-sized datasets: Statlog with 270 observations, Cleveland dataset with 297 observations, and Merged dataset with 918 observations. The sampling techniques synthetic minority over-sampling(SMOTE) and random over-sampler (ROS) were implemented to handle the imbalance in the datasets and the grid search cross-validation method was used to find optimal hyperparameters.
Project Diagram: 


![image](https://github.com/PatrykMprojects/Heart-disease-prediction-/assets/78304814/5957ecaf-df31-47f3-9566-8cfd1f72c253)

Created ensemble classifiers:

![image](https://github.com/PatrykMprojects/Heart-disease-prediction-/assets/78304814/bdb0ecfa-5f1a-4698-bfd8-97f02b5f6638)


Comparison research study results: 

When the datasets were split into 80% training and 20% testing: 

![image](https://github.com/PatrykMprojects/Heart-disease-prediction-/assets/78304814/2ef21d06-a554-4ed2-9b24-1d516c8ee93d)

When datasets were evaluated with Leave One Out Cross-validation method:

![image](https://github.com/PatrykMprojects/Heart-disease-prediction-/assets/78304814/21b0cafa-a00b-43cd-b4f7-fb44f6f9a730)

Results in the table: 
![image](https://github.com/PatrykMprojects/Heart-disease-prediction-/assets/78304814/9be81d80-198a-436c-9475-b4617486135e)
# TabPFN Web App for Heart Disease Prediction
# To run Web Application: 
To run the web app in Pycharm first app.py file have to be run:

![image](https://github.com/PatrykMprojects/Heart-disease-prediction-/assets/78304814/db45b586-89c6-46a5-a9ab-842b3136635e)

Then the command prompted above can be pasted into a terminal to display the web app: 
![image](https://github.com/PatrykMprojects/Heart-disease-prediction-/assets/78304814/df898991-80c0-474c-bca1-62f841e78eb5)

The web page should come up automatically. If not paste the URL that is given above. 

# How to use the Web Application: 
Users can provide their input in the boxes as displayed below: 

![image](https://github.com/PatrykMprojects/Heart-disease-prediction-/assets/78304814/cd26a6e4-6cd6-4f4b-9f85-dbecb9cfe681)
![image](https://github.com/PatrykMprojects/Heart-disease-prediction-/assets/78304814/3900936d-c433-412f-8d4b-bc7cc2a91c06)

Then after providing the patient details user can click on predict button and the result will display at the bottom as presented below. 

![image](https://github.com/PatrykMprojects/Heart-disease-prediction-/assets/78304814/9b4dfb0e-0a01-4eb4-a966-42e1257b93b1)
![image](https://github.com/PatrykMprojects/Heart-disease-prediction-/assets/78304814/992f7f4f-31f5-4499-b0ac-663364d7e9dd)

# Thanks 






