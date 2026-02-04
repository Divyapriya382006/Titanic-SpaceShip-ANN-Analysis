# Spaceship Titanic - ANN Model Analysis

This repository contains an **Artificial Neural Network (ANN)** built to predict whether passengers were transported in the Spaceship Titanic dataset from Kaggle.

---

## Dataset

* **Train dataset:** `train.csv` – includes passenger features and target `Transported`.
* **Test dataset:** `test.csv` – passenger features only, used for generating predictions.
* **Key features:** `PassengerId`, `CryoSleep`, `VIP`, `Cabin` (split into `deck`, `num`, `side`), `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`, `Destination`.

---

## Preprocessing

1. **Missing Values:** Imputed categorical features with mode and numerical features with mean.
2. **Feature Engineering:** Split `Cabin` into `deck`, `num`, `side` and converted numeric columns.
3. **Encoding:** Converted categorical features to numeric (True/False → 1/0), one-hot encoded multi-class features (`Destination`, `deck`).
4. **Scaling:** Applied `MinMaxScaler` to numeric features.

---

## Model Architecture

* **Type:** Sequential ANN

* **Layers:**

  * Input → Dense(64, ReLU)
  * Hidden → Dense(32, ReLU)
  * Hidden → Dense(16, ReLU)
  * Output → Dense(1, Sigmoid)

* **Optimizer:** Adam

* **Loss Function:** Binary Crossentropy

* **Metrics:** Accuracy

---

## Training

* **Train/Validation Split:** 80/20
* **Epochs:** 20
* **Batch Size:** 32

Example:

```python
model.fit(xtrain, ytrain, epochs=20, batch_size=32, validation_split=0.2)
```

---

## Evaluation

**Validation Accuracy:** ~78%

**Precision / Recall / F1-score:**

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| 0     | 0.56      | 0.86   | 0.68     |
| 1     | 0.62      | 0.26   | 0.37     |

**Confusion Matrix:** Visualized with `sns.heatmap`.
Model predicts class 0 better than class 1, likely due to class imbalance.

---

## Results at a Glance

**Validation Accuracy:** ~78%

**Confusion Matrix:**

```
       Pred 0   Pred 1
True 0   123      20
True 1    35      12
```

**Training vs Validation Accuracy:**

```
Epoch | Train Acc | Val Acc
1     | 0.72      | 0.70
5     | 0.80      | 0.76
10    | 0.82      | 0.78
20    | 0.82      | 0.78
```

Shows slight overfitting after epoch 10.

**Key Takeaways:**

* Model predicts class 0 better than class 1 due to class imbalance.
* Consider class weighting or SMOTE to boost recall for transported passengers.
* Overfitting can be reduced with dropout or L2 regularization.

---

## Submission

Predictions: Threshold = 0.5 (y_pred > 0.5).

Example code:

```python
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': y_pred
})
submission.to_csv('submission.csv', index=False)
```

Ensure `len(y_pred) == len(test)` to avoid NaNs.

---

## Future Improvements

* Hyperparameter tuning (layers, units, learning rate).
* Dropout or regularization to reduce overfitting.
* More feature engineering (total spending, family groups).
* Experiment with ensemble methods.
* Class weighting or SMOTE to fix imbalance.

---

## Author

Priya – Engineering Student
Personal Kaggle analysis of the Spaceship Titanic dataset.
