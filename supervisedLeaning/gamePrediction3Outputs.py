#input
number_fans = 50
#outputs
target_win = 0.6
target_draw = 0.3
target_lose = 0.1
#weights
weights = [0.7, 0.6, 11]

def neural_network(weights, input):
                #win draw lose
    prediction = [0 , 0, 0]
    prediction[0] = weights[0] * input
    prediction[1] = weights[1] * input
    prediction[2] = weights[2] * input
    return prediction

ai_pred =  neural_network(weights, input=number_fans)
print("AI PREDICTED :" + str(ai_pred))
