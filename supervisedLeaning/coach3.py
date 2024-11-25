input = 2
target_Prediction = 0.8
weight = 0.5
lr = 0.1


def neural_network(myweights):
    pred = myweights * input
    return pred


for i in range(20):
    ai_pred = neural_network(weight)
    error = (target_Prediction - ai_pred) ** 2  # mean squared errorr
    delta = ai_pred - target_Prediction
    weight_delta = delta * input  # how much weight caused us to miss
    weight = weight - weight_delta * lr
    print("Error is " + str(error))
    print("Delta:" + str(delta) + " WeightDelta:" + str(weight_delta))
