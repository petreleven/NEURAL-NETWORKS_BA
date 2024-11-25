input = 0.5
target_prediction = 0.75

# assume weight
weight = 20


def neuralNetwork(myweights):
    ai_pred = input * myweights
    return ai_pred


lr = 0.1
orig_pred = neuralNetwork(weight)
print("Original prediction is " + str(orig_pred))

for i in range(1000):
    ai_pred = neuralNetwork(weight)
    original_err = (target_prediction - ai_pred) ** 2
    # INCREASE WEIGHT
    ai_pred = neuralNetwork(weight + lr)
    inc_w_err = (target_prediction - ai_pred) ** 2
    # DECREASE WEIGHT
    ai_pred = neuralNetwork(weight - lr)
    dec_w_err = (target_prediction - ai_pred) ** 2
    # UPDATE WIGHT BY CHECKING LEAST ERROR
    if inc_w_err < original_err or dec_w_err < original_err:
        if inc_w_err < dec_w_err:
            weight = weight + lr
        elif dec_w_err < inc_w_err:
            weight = weight - lr

# HOT AND COLD LEARNING
# INEFFICIENT (3 times)
# 0.7 0.6 0.7 0.6
new_pred = neuralNetwork(weight)
print("New prediction is " + str(new_pred))
