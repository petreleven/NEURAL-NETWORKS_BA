input = 0.5
target_prediction_output = 0.75

#assume weight
weight = 20

def neuralNetwork():
    prediction = weight * input
    return prediction

ai_prediction = neuralNetwork()
error = (target_prediction_output - ai_prediction) ** 2 #mean squared error
print("ORIGINAL ERROR " +str(error))

#Learning Rate
lr = 0.1
weight = weight - lr
ai_prediction_with_increased_weight = neuralNetwork()
incr_weight_error = (target_prediction_output - ai_prediction_with_increased_weight)  ** 2
print("Increase Weight Error: " + str(incr_weight_error))
