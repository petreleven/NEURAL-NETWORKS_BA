input = 0.5
target_prediction = 0.75

# assume weight
weight = 20


def neuralNetwork(myinput, myweights):
    ai_pred = myinput * myweights
    return ai_pred


ai_pred = neuralNetwork(input, weight)
original_error = (target_prediction - ai_pred) ** 2  # MSE
print("ORIGINAL ERROR " + str(original_error))

# INCREASE WEIGHT
lr = 0.1
ai_pred = neuralNetwork(input, weight + lr)
new_error = (target_prediction - ai_pred) ** 2  # MSE
print("INCREASED WEIGHT ERROR " + str(new_error))

ai_pred = neuralNetwork(input, weight - lr)
new_error = (target_prediction - ai_pred) ** 2  # MSE
print("DECREASED WEIGHT ERROR " + str(new_error))
# UPDATE WEIGHT
weight = weight - lr
print("-" * 20)
# REPEATING
ai_pred = neuralNetwork(input, weight)
original_error = (target_prediction - ai_pred) ** 2  # MSE
print("ORIGINAL ERROR " + str(original_error))

ai_pred = neuralNetwork(input, weight + lr)
new_error = (target_prediction - ai_pred) ** 2  # MSE
print("INCREASED WEIGHT ERROR " + str(new_error))
