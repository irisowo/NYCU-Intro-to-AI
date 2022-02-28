import dataset
import adaboost
import utils
import detection
import matplotlib.pyplot as plt

import os
os.chdir('./Desktop/AI/Hw/HW1-code')

# Part 1: Implement loadImages function in dataset.py and test the following code.
print('Loading images')
trainData = dataset.loadImages('data/train')
print(f'The number of training samples loaded: {len(trainData)}')
testData = dataset.loadImages('data/test')
print(f'The number of test samples loaded: {len(testData)}')

print('Show the first and last images of training dataset')
'''
fig, ax = plt.subplots(1, 2)
ax[0].axis('off')
ax[0].set_title('Face')
ax[0].imshow(trainData[1][0], cmap='gray')
ax[1].axis('off')
ax[1].set_title('Non face')
ax[1].imshow(trainData[-1][0], cmap='gray')
plt.show()
'''

# Part 2: Implement selectBest function in adaboost.py and test the following code.
# Part 3: Modify difference values at parameter T of the Adaboost algorithm.
# And find better results. Please test value 1~10 at least.
train_log = []
test_log = []
last_T = 10
for T in range(1, (last_T + 1)):
    print('===================== T =',T,'=============================\n')
    print('Start training your classifier')
    clf = adaboost.Adaboost(T = T)
    clf.train(trainData)

    print('\nEvaluate your classifier with training dataset')
    train_false_positives, train_all_negatives, train_false_negatives, train_all_positives, train_correct = utils.evaluate(clf, trainData)
    train_log.append([train_false_positives, train_all_negatives, train_false_negatives, train_all_positives, train_correct])

    print('\nEvaluate your classifier with test dataset')
    test_false_positives, test_all_negatives, test_false_negatives, test_all_positives, test_correct = utils.evaluate(clf, testData)
    test_log.append([test_false_positives, test_all_negatives, test_false_negatives, test_all_positives, test_correct])
    print('\n')

    # Part 4: Implement detect function in detection.py and test the following code.
    if(T == last_T) :
        print('\nDetect faces at the assigned lacation using your classifier')
        detection.detect('data/detect/detectData.txt', clf)
    
    # Part 5: Test classifier on your own images
    '''
    if(T == last_T) :
        print('\nDetect faces on your own images')
        detection.detect('data/detect/MyOwnImages.txt', clf)
    '''
    
    print('\n')
