# Main.py
import subprocess
import sys
import os
from DataController import load_data, prepare_features
from modelTorch import train_model, predict_similarity, display_results, SimpleNN


class Main:
    def __init__(self, cv_path, vacature_path, labels_path, model_path='model.pth'):
        self.cv_path = cv_path
        self.vacature_path = vacature_path
        self.labels_path = labels_path
        self.model_path = model_path
        self.model = None

    def install_requirements(self):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully.")

    def load_model(self, input_size):
        model = SimpleNN(input_size)
        if os.path.exists(self.model_path):
            model.load_model(self.model_path)
        return model

    def run(self):
        # Install requirements
        self.install_requirements()

        # Load data from CSV files
        cv_df, vacature_df, labels_df = load_data(self.cv_path, self.vacature_path, self.labels_path)

        # Prepare features and labels for the model
        X, y = prepare_features(cv_df, vacature_df, labels_df)

        # Check if a saved model exists
        if os.path.exists(self.model_path):
            self.model = self.load_model(X.shape[1])
        else:
            # Train the model
            self.model, X_test, y_test = train_model(X, y, self.model_path)

        # Predict similarity scores
        predictions = predict_similarity(self.model, X)

        # Display the prediction results
        display_results(cv_df, vacature_df, labels_df, predictions, self.model, X, y, learning_rate=0.001)


if __name__ == "__main__":
    cv_path = 'data/cv_data.csv'
    vacature_path = 'data/vacature_data.csv'
    labels_path = 'data/labels.csv'

    main = Main(cv_path, vacature_path, labels_path)
    main.run()
