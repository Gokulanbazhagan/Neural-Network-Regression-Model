# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Developing a neural network regression model involves designing a feedforward neural network with fully connected layers to predict continuous numerical values. The model is trained by minimizing a loss function such as Mean Squared Error (MSE), which quantifies the difference between predicted and actual values. Optimization algorithms like RMSprop or Adam are used to adjust the model's weights during training, enabling it to learn patterns in the data and improve its predictions over time.

## Neural Network Model

![image](https://github.com/user-attachments/assets/3406031d-4fec-4a87-8b24-c76620035fd3)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Gokularamanan K
### Register Number: 212222230040
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.001)



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```
## Dataset Information

![image](https://github.com/user-attachments/assets/b81c2844-f926-4fc1-b38a-5d02233d81d7)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2025-02-27 143910](https://github.com/user-attachments/assets/337adeee-d337-4df2-8004-9d964de56664)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/fa26358e-38ed-4322-8a17-b77294314588)


## RESULT
A neural network regression model predicts continuous values by minimizing a loss function like Mean Squared Error (MSE) using optimization algorithms such as RMSprop or Adam.
