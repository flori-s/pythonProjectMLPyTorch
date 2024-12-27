from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate

from DataController import prepare_features


# Define the Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First layer
        self.fc2 = nn.Linear(128, 64)  # Second layer
        self.fc3 = nn.Linear(64, 1)  # Output layer
        self.relu = nn.ReLU()
        self.output = nn.Sigmoid()  # For output between 0 and 1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.output(x)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
        print(f"Model loaded from {path}")


# Function to train the model
def train_model(X, y, epochs=100, learning_rate=0.001):
    # Convert data to PyTorch tensors
    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Initialize the model, loss function, and optimizer
    model = SimpleNN(input_size=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    model.save_model('model.pth')

    return model, X_test, y_test


# Function to predict similarity scores using the trained model
def predict_similarity(model, X):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X.values))
    return predictions.numpy()


# Function to display the prediction results using tabulate
def display_results(cv_df, vacature_df, labels_df, predictions, model, X, y, learning_rate=0.001):
    headers = ["Name", "Vacancy Title", "Predicted Match Percentage"]
    table = []

    for i, row in labels_df.iterrows():
        cv_id = int(row['cv_id']) - 1  # Adjust index to 0-based
        vacature_id = int(row['vacature_id']) - 1  # Adjust index to 0-based
        cv_name = cv_df.iloc[cv_id]['Naam']  # Replace 'Naam' with the appropriate column for the CV name
        vacature_title = vacature_df.iloc[vacature_id]['Functietitel']  # Replace 'Functietitel' as needed
        percentage = predictions[i] * 100  # Convert to percentage
        table.append((cv_name, vacature_title, percentage))

    # Sort the table by the match percentage in descending order
    table.sort(key=lambda x: x[2], reverse=True)

    # Group by name
    grouped_table = {}
    for row in table:
        name = row[0]
        if name not in grouped_table:
            grouped_table[name] = []
        grouped_table[name].append(row)

    # Print the grouped results and get feedback
    for name, rows in grouped_table.items():
        print(f"\nResults for {name}:")
        print(tabulate(rows, headers, tablefmt="grid"))
        for row in rows:
            feedback = input(f"Is the match for {row[1]} correct? (yes/no): ").strip().lower()
            if feedback == 'no':
                # Update the model with the correct feedback
                correct_score = float(input(f"Enter the correct match percentage for {row[1]}: ")) / 100
                update_model(model, X, y, row, correct_score, learning_rate, cv_df, vacature_df, labels_df)


def update_model(model, X, y, row, correct_score, learning_rate, cv_df, vacature_df, labels_df):
    # Find the index of the row in the original data
    cv_name, vacature_title, _ = row
    cv_id = cv_df[cv_df['Naam'] == cv_name].index[0]
    vacature_id = vacature_df[vacature_df['Functietitel'] == vacature_title].index[0]
    index = labels_df[(labels_df['cv_id'] == cv_id + 1) & (labels_df['vacature_id'] == vacature_id + 1)].index[0]

    # Update the label with the correct score
    labels_df.at[
        index, 'similarity_score'] = correct_score  # Assuming 'similarity_score' is the column name for the labels

    # Save the updated labels DataFrame to CSV
    labels_df.to_csv('data/labels.csv', index=False)

    # Re-prepare features and labels
    X, y = prepare_features(cv_df, vacature_df, labels_df)

    # Convert data to PyTorch tensors
    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    # Initialize the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Update the model with the new data
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Model updated with the correct score for {vacature_title}")
