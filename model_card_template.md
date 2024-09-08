# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was created as part of a machine learning pipeline built with FastAPI for scalability. It predicts an individual's income level based on various demographic attributes like age, workclass, education, marital status, occupation, relationship, race, gender, and country of origin.

## Intended Use
The primary purpose of this model is to serve as a learning tool, illustrating the steps involved in building and deploying a machine learning pipeline. It is not suitable for real-world decision-making without further refinement and validation.

## Training Data
The model was trained using the census.csv file, which holds demographic details of individuals. A train-test split was applied to divide the data into training and testing sets, with categorical variables processed through one-hot encoding.

## Evaluation Data
The evaluation of the model was done using a subset of the census.csv dataset that was reserved for testing. The test data underwent the same preprocessing steps as the training data to maintain consistency.

## Metrics
The model's performance were evaluated based on the following metrics on the test data:
* Precision: 0.2407
* Recall: 1.0000
* F1 Score: 0.3881
These metrics were calculated based on the model's predictions on the test dataset and reflect the model's ability to correctly identify income categories based on its predictions.

## Ethical Considerations
Since this model was developed for educational purposes, it should not be applied in production environments or used in decision-making processes that affect people. The model lacks critical fairness and accuracy checks, and potential biases in demographic-based predictions could occur and should be carefully considered.

## Caveats and Recommendations
* The model exhibits high recall but suffers from low precision, meaning it successfully identifies positive cases but produces many false positives.
* Tuning the model further, such as by testing alternative algorithms or adjusting hyperparameters, may help balance precision and recall.
* Testing the model on additional datasets is recommended to verify its ability to generalize across different data.
* The ethical risks of using demographic data in predictive modeling should be thoroughly addressed before any deployment in real-world scenarios.