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
```
# Name: Gokularamanan K
# Reg No:212222230040
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


```
```

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)
ai_brain.history = {'loss': []}

```

```

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=1000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```
## Dataset Information

![Screenshot 2025-03-09 120230](https://github.com/user-attachments/assets/4f666bfc-0cb6-48f4-997c-5a532c142fcd)



## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2025-03-09 115844](https://github.com/user-attachments/assets/4e289032-dfd7-4f3e-9ec8-1cd89016d7f6)



### New Sample Data Prediction

![Screenshot 2025-03-09 115947](https://github.com/user-attachments/assets/571b1c92-4783-4930-8b7b-f670d64e767e)



## RESULT
A neural network regression model predicts continuous values by minimizing a loss function like Mean Squared Error (MSE) using optimization algorithms such as RMSprop or Adam.
