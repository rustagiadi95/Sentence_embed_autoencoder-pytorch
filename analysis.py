import matplotlib.pyplot as plt

f = open('logs.txt', 'r')
li = [items for items in f.readlines()]
training = [li[items].strip('\n') for items in range(1, len(li), 4)]
validation = [li[items].strip('\n') for items in range(2, len(li), 4)]
training_losses = []
training_accuracy = []
validation_losses = []
validation_accuracy = []
for items in range(len(training)) :
    li = training[items].split(',')
    training_losses.append(float(li[1].strip(' Training Loss: ')))
    training_accuracy.append(float(li[2].strip(' Training Accuracy: ')))

    li = validation[items].split(',')
    validation_losses.append(float(li[1].strip(' Validation Loss: ')))
    validation_accuracy.append(float(li[2].strip(' Validation Accuracy: ')))

fig, ax = plt.subplots(2, 1, figsize = (10, 11))
ax[0].plot([items for items in range(len(training))], training_losses, color='b', label='Training')
ax[0].plot([items for items in range(len(training))], validation_losses, color='r', label='Validation')
ax[0].set_title('Losses vs Epochs')
ax[0].legend(loc="upper right")
ax[0].set_xticks([items for items in range(len(training))])
ax[0].set_xticklabels([items for items in range(len(training))], fontsize=8)
ax[0].grid()

ax[1].plot([items for items in range(len(training))], training_accuracy, color='b', label='Training')
ax[1].plot([items for items in range(len(training))], validation_accuracy, color='r', label='Validation')
ax[1].set_title("Accuracy vs Epochs")
ax[1].legend(loc="lower right")
ax[1].set_xticks([items for items in range(len(training))])
ax[1].set_xticklabels([items for items in range(len(training))], fontsize=8)
ax[1].grid()
plt.show()