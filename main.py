import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the csv file
data = pd.read_csv('data2.csv')

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].SAT
        y = points.iloc[i].GPA
        total_error += abs(y - (m * x + b))
    return total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].SAT
        y = points.iloc[i].GPA
        m_gradient += -(2 / float(n)) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / float(n)) * (y - (m_now * x + b_now))

    new_m = m_now - (L * m_gradient)
    new_b = b_now - (L * b_gradient)
    
    return new_m, new_b

def predict_GPA(SAT, m, b):
    return m * SAT + b

m = 0
b = 0
L = 0.0000001
epochs = 500

# Training the model
for i in range(epochs):
    if i % 100 == 0:
        # print(loss_function(m, b, data))
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, L)

print("Final m:", m)
print("Final b:", b)

# Plotting the data and the regression line
plt.scatter(data.SAT, data.GPA)
plt.plot(data.SAT, m * data.SAT + b, color='red')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.title('Linear Regression Fit')
plt.show()

# Allowing the user to make predictions
while True:
    try:
        SAT_input = input("Enter an SAT to predict GPA (or enter 'exit' to stop): ")
        if SAT_input == 'exit':
            break
        predicted_GPA = predict_GPA(float(SAT_input), m, b)
        print(f"Predicted GPA for SAT {SAT_input}: {predicted_GPA:.2f}")
    except ValueError:
        print("Invalid input. Please enter a valid numeric SAT.")
