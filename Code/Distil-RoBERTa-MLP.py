# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 00:16:56 2024

@author: Lenovo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
from lime.lime_text import LimeTextExplainer

# Load your dataset (CSV file)
data = pd.read_excel('Firefox-dataset.xlsx')

data['Part1']= data["Title1"] + data["Description1"] + data["steps1"] + data["actual1"]+ data["expected1"]
data['Part2']= data["Title2"] + data["Description2"] + data["steps2 "] + data["actual2"]+ data["expected2"]

X = data[['bug_id1','actual1', 'bug_id2','actual2','actual_cosine_score']]#'steps_cosine_score'
y = data['Label']


# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Load the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

# Encode sentence pairs
X_emb = model.encode(list(zip(X['Title1'], X['Title2'])), convert_to_tensor=True )
#X_emb = model.encode(list(zip(X['Desc1'], X['Desc2'])), convert_to_tensor=True )
#X_emb = model.encode(list(zip(X['S2R1'], X['S2R2'])), convert_to_tensor=True )
#X_emb = model.encode(list(zip(X['AR1'], X['AR2'])), convert_to_tensor=True )
#X_emb = model.encode(list(zip(X['ER1'], X['ER2'])), convert_to_tensor=True )

import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768 * 2, 768)  # Adjust the input dimension to match MLP input

    def forward(self, sentences_pair):
        embeddings = self.model.encode(sentences_pair, convert_to_tensor=True)
        pooled_output = self.dropout(embeddings)
        features = self.fc(pooled_output)
        return features

# Initialize the feature extractor and the MLP classifier
feature_extractor = FeatureExtractor()

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
mlp_classifier = MLPClassifier(max_iter=100)

# Define a custom prediction function for Lime: takes one single instance
def predict_with_transformer_mlp(sentences_pair, model, classifier, fold):
    # Encode sentence pair
    text_embeddings = model.encode(sentences_pair, convert_to_tensor=True)
    text_embeddings= text_embeddings.cpu().numpy()

    # Make predictions using the MLP classifier and return probabilities
    #prediction_probs = classifier['estimator'][fold].predict_proba(text_embeddings)
    prediction_probs = classifier.predict_proba(text_embeddings)


    return prediction_probs

# Initialize LimeTextExplainer
explainer = LimeTextExplainer(class_names=["Non-Duplicate", "Duplicate"])

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Loop over folds
for fold, (train_index, test_index) in enumerate(kf.split(X_emb, y)):
    X_train, X_test = X_emb[train_index], X_emb[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model on the training set
    mlp_classifier.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = mlp_classifier.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print or store the evaluation metrics for each fold
    print(f"Fold {fold + 1} Evaluation Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print("-------------------------")

    # Now we can use the 'predict_with_transformer_mlp' function with LimeTextExplainer:
    #fold_prediction_func = lambda x: predict_with_transformer_mlp(x, model, mlp_classifier, fold)

    # Choose a random instance for explanation
    #sample_index = test_index[0]  # Using the first instance from the test set of the current fold
    #sample_text_a = data['Title1'][sample_index]
    #sample_text_b = data['Title2'][sample_index]
    #sample_text_pair = [sample_text_a, sample_text_b]
    #sample_text_pair=f"{sample_text_a} {sample_text_b}"


    # Generate an explanation for the selected instance
    #explanation = explainer.explain_instance(
        #sample_text_pair, fold_prediction_func, num_features=10, top_labels=1
   # )

    # Display the explanation
   # print(f"Fold {fold + 1} Explanation:")
   # explanation.show_in_notebook()
   
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix,recall_score, f1_score, confusion_matrix, matthews_corrcoef
# Initialize variables to store cumulative metrics
avg_accuracy = 0.0
avg_precision = 0.0
avg_recall = 0.0
avg_f1 = 0.0

# Initialize variables to store cumulative metrics
avg_specificity = 0.0
avg_mcc = 0.0


# Initialize a DataFrame to store prediction samples
predictions_df = pd.DataFrame(columns=['Fold','SampleIndex','ActualLabel', 'PredictedLabel','actual_cosine_score'])#'TextSample'
misclassifications_df = pd.DataFrame(columns=['Fold','MisclassifiedIndex', 'ActualLabel', 'PredictedLabel','actual_cosine_score'])#'TextSample',
full_misclassifications_df = pd.DataFrame(columns=['Fold', 'MisclassifiedIndex', 'Issue_id1','BR1','Issue_id2','BR2', 'ActualLabel', 'PredictedLabel', 'actual_cosine_score'])#'TextSample',


# Loop over folds
for fold, (train_index, test_index) in enumerate(kf.split(X_emb_cpu, y)):
    X_train, X_test = X_emb_cpu[train_index], X_emb_cpu[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model on the training set
    mlp_classifier.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = mlp_classifier.predict(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Save prediction samples to the DataFrame
    fold_samples = pd.DataFrame({
        'Fold': [fold + 1] * len(test_index),
        'SampleIndex': test_index,
        #'BR1': X.iloc[test_index]['Title1'],
        #'BR2': X.iloc[test_index]['Title2'],#+ ' ' + ,
        'ActualLabel': y[test_index],
        'PredictedLabel': y_pred,
        'actual_cosine_score': X.iloc[test_index]['actual_cosine_score'],

    })

    misclassified_indices = test_index[y_test != y_pred]

    if not misclassified_indices.size:
    # Skip the calculations for this fold if there are no misclassifications
       continue

    misclassifications = pd.DataFrame({
    'Fold': [fold + 1] * len(misclassified_indices),
    'MisclassifiedIndex': misclassified_indices,
    'ActualLabel': pd.Series(y).iloc[misclassified_indices].tolist(),
    'PredictedLabel': y_pred[y_test != y_pred].tolist(),
    'actual_cosine_score': X.iloc[misclassified_indices]['actual_cosine_score'].tolist(),
     })

    full_misclassifications = pd.DataFrame({
    'Fold': [fold + 1] * len(misclassified_indices),
    'MisclassifiedIndex': misclassified_indices,
    'Issue_id1': X.iloc[misclassified_indices]['bug_id1'].tolist(),
    'BR1': X.iloc[misclassified_indices]['actual1'].tolist(),
    'Issue_id2': X.iloc[misclassified_indices]['bug_id2'].tolist(),
    'BR2': X.iloc[misclassified_indices]['actual2'].tolist(),
    'ActualLabel': pd.Series(y).iloc[misclassified_indices].tolist(),
    'PredictedLabel': y_pred[y_test != y_pred].tolist(),
    'actual_cosine_score': X.iloc[misclassified_indices]['actual_cosine_score'].tolist(),
    })


    predictions_df = pd.concat([predictions_df, fold_samples], ignore_index=True)
    misclassifications_df = pd.concat([misclassifications_df, misclassifications], ignore_index=True)
    full_misclassifications_df = pd.concat([full_misclassifications_df, full_misclassifications], ignore_index=True)

    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    # Calculate MCC
    mcc = matthews_corrcoef(y_test, y_pred)

    # Accumulate metrics
    avg_specificity += specificity
    avg_mcc += mcc



    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Accumulate metrics
    avg_accuracy += accuracy
    avg_precision += precision
    avg_recall += recall
    avg_f1 += f1


    # Print or store the evaluation metrics for each fold
    print(f"Fold {fold + 1} Evaluation Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"Specificity: {specificity}")
    print(f"MCC: {mcc}")
    print(f"Misclassifications: {len(misclassified_indices)}")
    # Print confusion matrix
    print(f"Confusion Matrix - Fold {fold + 1}:")
    print(cm)
    print("-------------------------")

    # Now we can use the 'predict_with_transformer_mlp' function with LimeTextExplainer:
    fold_prediction_func = lambda x: predict_with_transformer_mlp(x, model, mlp_classifier, fold)

    # Now we can use the 'predict_with_transformer_mlp' function with LimeTextExplainer:
    #fold_prediction_func = lambda x: predict_with_transformer_mlp(x, model, mlp_classifier, fold)

    # Choose a random instance for explanation
    sample_index = test_index[0]  # Using the first instance from the test set of the current fold
    sample_text_a = "None of them workw" #data['steps1'][sample_index]
    sample_text_b = "Menus do not display. The menus border becomes pressed but no menu appears below the title"#data['steps2'][sample_index]
    #sample_text_pair = [sample_text_a, sample_text_b]
    sample_text_pair=f"{sample_text_a} {sample_text_b}"


    # Generate an explanation for the selected instance
    explanation = explainer.explain_instance(
        sample_text_pair, fold_prediction_func, num_features=10, top_labels=1 )

    # Display the explanation
    print(f"Fold {fold + 1} Explanation:")
    explanation.show_in_notebook()


# Save the prediction samples to a CSV file
#predictions_df.to_excel('@[AR]_prediction_samples-Fx.xlsx', index=False)
#misclassifications_df.to_excel('@[AR]_misclassifications-Fx.xlsx', index=False)
#full_misclassifications_df.to_excel('@[AR]_Full-misclassifications-Fx.xlsx', index=False)





# Calculate average metrics across all folds
avg_accuracy /= kf.get_n_splits()
avg_precision /= kf.get_n_splits()
avg_recall /= kf.get_n_splits()
avg_f1 /= kf.get_n_splits()

# Calculate average metrics across all folds
avg_specificity /= kf.get_n_splits()
avg_mcc /= kf.get_n_splits()

# Print average metrics
print("Average Performance Metrics:")
print(f"Average Accuracy: {avg_accuracy}")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average F1-score: {avg_f1}")
print(f"Average Specificity: {avg_specificity}")
print(f"Average MCC: {avg_mcc}")

   
