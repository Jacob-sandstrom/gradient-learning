import gym
import random
import numpy as np
import time
# import yaml

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def max_index(array):
    max_i = 0
    i = 1
    while (i < len(array)):
        if (array[i] > array[max_i]):
            max_i = i
        i += 1
    return max_i

def mean(array):
    total = 0
    for num in array:
        total += num
    mean = total / len(array)
    return mean

class Network:

    def __init__(self, sizes):
        self.layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.learn_rate = 0.01
        self.gamma = 0.99

        # print(self.weights)
        
        
    def feedforward(self, a):
        values = [a]
        all_z = [[None]]
        for w in self.weights:
            z = np.dot(w, a)
            all_z.append(z)
            a = sigmoid(z)
            values.append(a)
        return values, all_z

    def backprop(self, all_n_values, z_values, all_w, actions, desiered_val):

        change_w = [np.zeros(w.shape) for w in self.weights]

        steps_from_start = 0

        # for val in all_n_values:
        #     l = 1
        #     if desiered_val == 1:
        #         change_v[-1][a][0] = 1
        #     else:
        #         for index in range(len(change_v[-1])):
        #             change_v[-1][index][0] = 1
        #         change_v[-1][a][0] = 0
            
        #     while l < self.layers:
        #         for i in range(len(val[-l])):
        #             for j in range(len(w[-l][i])):
                        
        #         l += 1
        #     steps_from_start += 1


        # steps_from_start = 0

        # change_v = [np.zeros(v.shape) for v in all_n_values[0]]
        # change_v[-l-1][j] -= (w[-l][i][j] * sigmoid_prime(z_val[-l][i]) * 2 * (val[-l][i] - change_v[-l][i][0])) * (self.gamma**steps_from_start) * self.learn_rate


        #   network
        for val, z_val, w, a in zip(all_n_values, z_values, all_w, actions):
            l = 1
            change_v = [np.zeros(v.shape) for v in all_n_values[0]]
            # print(a)
            if desiered_val == 1:
                change_v[-1][a][0] = 1
            else:
                for index in range(len(change_v[-1])):
                    change_v[-1][index][0] = 1
                change_v[-1][a][0] = 0

            #   layer
            while l < self.layers:

                #   neuron
                for i in range(len(val[-l])):
                    #   weights connected to specific neuron
                    #   j is a specific weight and also the neuron it is connected to in the previous layer
                    for j in range(len(w[-l][i])):
                        change_w[-l][i][j] -= (val[(-l-1)][j] * sigmoid_prime(z_val[-l][i]) * 2 * (val[-l][i] - change_v[-l][i][0])) * (self.gamma**steps_from_start) * self.learn_rate
                        change_v[-l-1][j] -= (w[-l][i][j] * sigmoid_prime(z_val[-l][i]) * 2 * (val[-l][i] - change_v[-l][i][0])) * (self.gamma**steps_from_start) * self.learn_rate

                l += 1
            steps_from_start += 1
        #   increase/decrease each weight by calculated value
        for layer in range(len(self.weights)):
            for neuron in range(len(self.weights[layer])):
                for weight in range(len(self.weights[layer][neuron])):
                    self.weights[layer][neuron][weight] += change_w[layer][neuron][weight]

        # print(change_w)



env = gym.make('CartPole-v1')
# env = gym.make('MountainCar-v0')
env.reset()
observation = env.reset()


game_runsteps = 10000
traning_games = 100000
games_to_show = 1
#   get initial score to beat
scores_to_collect = 10000
scores = []
for _ in range(scores_to_collect):
    env.reset()
    score = 0
    for _ in range(game_runsteps):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        score += reward
        if (done):
            break
    scores.append(score)
env.reset()
env.close()


score_to_beat = mean(scores)
print("score to beat:")
print(score_to_beat)



start = time.time()



observation, reward, done, info = env.step(action)



input_layer = len(observation)

network = Network([input_layer, 16, 16, env.action_space.n])
for i in range(traning_games):
    env.reset()
    score = 0 
    n_values_game = []
    weights_game = []
    actions = []
    z_values = []
    for _ in range(game_runsteps):
        # if i % games_to_show == 0:
        # env.render()
        n_values, z = network.feedforward(np.reshape(observation, (len(observation), 1)))
        result = n_values[-1]
        r = np.reshape(result, len(result))
        action = max_index(r)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        score += reward
        #   store values for backprop
        actions.append(action)
        z_values.append(z)
        n_values_game.append(n_values)
        weights_game.append(network.weights)
        if (done):
            scores.append(score)
            score_to_beat = mean(scores)
            # print(score)
            #   check if score is over desiered amount
            #   The last input to the function is the desiered output for the output neuron which was triggered
            #   If the network preformed poorly similar actions will be discouraged but if it preformed well the actions taken will be encouraged
            if score > score_to_beat:
                network.backprop(n_values_game, z_values, weights_game, actions, 1)
            # else:
                # network.backprop(n_values_game, z_values, weights_game, actions, 0)
            break
    if i % 500 == 0:
        print("iteration: " + str(i))
        print("score = " + str(score))
        print("mean = " + str(mean(scores)))
    
env.reset()
env.close()

print(score_to_beat)
# for _ in range(1):
#     result = network.feedforward(np.reshape(observation, (len(observation), 1)))


end = time.time()
print(end - start)








# parsed = yaml.load(open('weights.yaml', 'r'), Loader=yaml.FullLoader)
# print(parsed)


# file = open('weights.yaml', 'w')
# yaml.dump(array, file)
# file.close()

# np.savetxt("weights.txt", self.weights)
# np.savetxt("weights.txt", self.weights, fmt='%s')

# x = np.loadtxt("weights.txt")
# print(x)
# file = open('weights.yaml', 'w')
# yaml.dump(self.weights, file)
# file.close()
