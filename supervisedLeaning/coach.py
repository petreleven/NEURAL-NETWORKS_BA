input = 0.5
target_prediction_output = 0.75

# assume weight
weight = 20


def neuralNetwork():
    prediction = weight * input
    return prediction


ai_prediction = neuralNetwork()
print("AI PREDDICTED " + str(ai_prediction))
error = (target_prediction_output - ai_prediction) ** 2  # Mean squared error
print(error)
