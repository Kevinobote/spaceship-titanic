# Spaceship Titanic Prediction

## Project Overview
This project involves predicting whether passengers aboard the Spaceship Titanic were transported to an alternate dimension during a collision with a spacetime anomaly. The dataset contains records of about 13,000 passengers, and the task is to predict the 'Transported' status for the remaining passengers.

## Project Structure
- `train.csv`: Training dataset with features and target labels.
- `test.csv`: Test dataset with features, but without target labels.
- `sample_submission.csv`: Sample submission file format.
- `spaceship_titanic_prediction.ipynb`: Jupyter Notebook containing the project code.
- `README.md`: Documentation of the project.

## Installation and Setup
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd spaceship-titanic
   ```

2. **Install necessary packages**:
   Ensure you have Python 3.7+ and install the required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Load the datasets**:
   Place the `train.csv` and `test.csv` files in the project directory.

## Data Description
### Features:
- `PassengerId`: Unique ID for each passenger (format: gggg_pp).
- `HomePlanet`: Planet the passenger departed from.
- `CryoSleep`: Indicates if the passenger was in cryosleep.
- `Cabin`: Cabin number (format: deck/num/side).
- `Destination`: Planet the passenger is traveling to.
- `Age`: Age of the passenger.
- `VIP`: If the passenger is a VIP.
- `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`: Amount billed at each amenity.
- `Name`: First and last names of the passenger.
- `Transported`: Target variable indicating if the passenger was transported (True/False).

### Target:
- `Transported`: Boolean indicating if the passenger was transported to another dimension.

## Data Preprocessing
1. **Handling Missing Values**:
   - Used `SimpleImputer` with strategy 'most_frequent' for categorical features.
   - Imputed numerical features with their mean values.

2. **Encoding Categorical Variables**:
   - Applied `LabelEncoder` to convert categorical features to numeric.

3. **Feature Selection**:
   - Dropped `Name` and `PassengerId` as they do not provide predictive value.

## Exploratory Data Analysis (EDA)
- Visualized distribution of the target variable.
- Analyzed distributions of numerical features.
- Investigated relationships between features and the target variable.

## Model Training and Evaluation
1. **Model Selection**:
   - Used `RandomForestClassifier` for its robustness and ease of use.

2. **Training and Validation**:
   - Split the training data into train and validation sets (80-20 split).
   - Trained the model on the training set and evaluated on the validation set.
   - Used cross-validation to ensure robust evaluation.

3. **Performance Metrics**:
   - Evaluated model performance using accuracy.
   - Achieved a validation accuracy of ~0.80.

## Prediction and Submission
1. **Predicting Test Set**:
   - Used the trained model to predict the 'Transported' status for the test set.

2. **Preparing Submission File**:
   - Created a submission file with columns `PassengerId` and `Transported`.

## Running the Code
1. Open the `spaceship_titanic_prediction.ipynb` notebook.
2. Run the cells sequentially to perform data preprocessing, model training, evaluation, and prediction.
3. The final submission file `submission.csv` will be generated in the project directory.

## Conclusion
This project demonstrates the application of data preprocessing, exploratory data analysis, and machine learning techniques to predict the outcome of a futuristic scenario. The model built can predict whether passengers were transported to another dimension with reasonable accuracy.

## Future Work
- Experiment with different machine learning algorithms and ensemble methods.
- Fine-tune hyperparameters for improved performance.
- Explore additional feature engineering techniques.