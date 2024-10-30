#OUR INPUTS
toes = 10
previous_wins = 20
number_fans = 15
target = 0.8 #80% chance of winning
lr = 0.1

#OUR WEIGHTS
         #toes prvWins #fans
weights = [0.1, 0.2,  -0.1]
inputs = [toes, previous_wins, number_fans]

def neuralNetwork(weights):
    prediction = 0
    for i in range(len(inputs)):
        single_pred = inputs[i]  * weights[i]
        prediction = prediction + single_pred

    return prediction

def calculateWeightDeltas(delta : float ,  inputs : list):
    weights_delta = [0, 0, 0]
    for i in range(len(inputs)):
        single_w_delta = inputs[i] * delta
        weights_delta[i] = single_w_delta * lr

    return weights_delta


ai_pred = neuralNetwork(weights)
print("PREDICTION IS :" + str(ai_pred) + " TARGET IS " + str(target))
error = (ai_pred - target) ** 2
print("Error is :" + str(error))
delta = (ai_pred - target)
weights_delta = calculateWeightDeltas(delta, inputs)
print("Weight deltas :"+ str(weights_delta))
