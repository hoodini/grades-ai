import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Generate a more realistic dataset with 100K entries
np.random.seed(0)
ages = np.random.randint(18, 60, 100000)
genders = np.random.randint(0, 2, 100000)
sleep_hours = np.random.randint(4, 12, 100000)

# A more complex formula to generate the labels
labels = ((100 - np.abs(ages - 25) * 0.3) + 
          (genders * 10) - 
          (np.abs(sleep_hours - (6 + (ages / 20))) * 5) + 
          np.random.randn(100000) * 5)

# Keep the labels in the range of 0 to 100
labels = np.clip(labels, 0, 100)

data = np.vstack((ages, genders, sleep_hours)).T

# Split data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Scale your features
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def train(self, data, labels, val_data, val_labels, epochs=100, lr=0.0001, batch_size=100):
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = len(data) // batch_size

            for i in range(num_batches):
                batch_data = data[i*batch_size:(i+1)*batch_size]
                batch_labels = labels[i*batch_size:(i+1)*batch_size]

                grad_w1 = np.zeros_like(self.w1)
                grad_w2 = np.zeros_like(self.w2)
                batch_loss = 0

                for x, y_true in zip(batch_data, batch_labels):
                    x = x.reshape(-1, 1)
                    y_true = y_true.reshape(-1, 1)

                    h = self.relu(self.w1.T.dot(x))
                    y_pred = self.w2.T.dot(h)

                    loss = np.mean((y_pred - y_true) ** 2)
                    batch_loss += loss

                    grad_y_pred = 2.0 * (y_pred - y_true)
                    grad_w2 += h.dot(grad_y_pred.T)
                    grad_h = self.w2.dot(grad_y_pred)
                    grad_w1 += x.dot((grad_h * (h > 0)).T)

                self.w1 -= lr * (grad_w1 / batch_size)
                self.w2 -= lr * (grad_w2 / batch_size)
                epoch_loss += batch_loss / batch_size

            val_loss = self.calculate_loss(val_data, val_labels)
            print(f'Epoch {epoch+1}, Loss: {epoch_loss/num_batches}, Val Loss: {val_loss}')

    def calculate_loss(self, data, labels):
        total_loss = 0
        for x, y_true in zip(data, labels):
            x = x.reshape(-1, 1)
            y_true = y_true.reshape(-1, 1)
            
            h = self.relu(self.w1.T.dot(x))
            y_pred = self.w2.T.dot(h)
            
            loss = np.mean((y_pred - y_true) ** 2)
            total_loss += loss
        
        return total_loss / len(data)

# Initialize and train the model
model = SimpleNN(3, 10, 1)
model.train(train_data, train_labels, val_data, val_labels)

# Save the trained model weights
np.save('w1.npy', model.w1)
np.save('w2.npy', model.w2)
