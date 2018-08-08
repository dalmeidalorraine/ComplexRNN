def getData(x):
    seq = []
    next_val = []
    for i in range(0, len(x) - steps_of_history - steps_in_future, steps_in_future):
        seq.append(x[i: i + steps_of_history])
        next_val.append(x[i + steps_of_history + steps_in_future -1])
    
    seq = np.reshape(seq, [-1, steps_of_history, 1])
    next_val = np.reshape(next_val, [-1, 1])
    X = np.array(seq)
    Y = np.array(next_val)
    return X,Y

step_radians = 0.01
steps_of_history = 20
steps_in_future = 1
learning_rate = 0.003
trainingData = X_train_scaled[:,0]
x_train,y_train = getData(trainingData)


TS = np.array(x_train)
n_steps = 20
f_horizon = 1

x_data = TS[:(len(TS) - (len(TS) % n_steps))]
x_batches = x_data.reshape(-1, 20, 1)

y_data = TS[1:(len(TS) - (len(TS) % n_steps))+f_horizon]
y_batches = y_data.reshape(-1, 20, 1)

print(x_batches.shape)
print(y_batches.shape)