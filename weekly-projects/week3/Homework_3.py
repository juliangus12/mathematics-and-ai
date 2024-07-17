import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Part 1: Data preparation

# toggle settings
add_noise = False

# Initialize lists for image collection
train_images = []
test_images = []

for i, images in enumerate([train_images, test_images]):
    # set paths to images of pandas and bears in train and test set
    datasetname = ['Train', 'Test'][i]
    folder_path1 = f'weekly-projects/week3/PandasBears/{datasetname}/Pandas/'
    folder_path2 = f'weekly-projects/week3/PandasBears/{datasetname}/Bears/'

    for folder_path in [folder_path1, folder_path2]:
        print(folder_path, end=' ')

        file_count = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                file_count += 1

                file_path = os.path.join(folder_path, filename)

                image = plt.imread(file_path, format='jpeg')

                image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

                image = image[::2, ::2]

                if add_noise:
                    image = image + np.random.normal(scale=100, size=image.shape)

                images.append(image)

        print('has {} images'.format(file_count))

# look at 4 random bears
plt.figure(figsize=(10, 10))
for i0, i in enumerate(np.random.randint(0, len(train_images), size=4)):
    plt.subplot(2, 2, 1 + i0)
    plt.imshow(train_images[i], cmap='Greys_r')
plt.show()

# Part 2: Singular value decomposition

# Follow the steps in the eigenfaces tutorial from the UW databook to perform an SVD on the images.
# Note that you will need to "center the data" before doing the SVD. Data centering means replacing each variable
# with a new variable that is equal to the variable minus its mean value. 
# (Think carefully about whether you want to use the mean of train set, test set, or the full data set for this.)

# Construct data matrix of centered data
all_images = np.array(train_images + test_images)
mean_image = np.mean(all_images, axis=0)  # Using the mean of the full dataset
centered_images = all_images - mean_image
A = centered_images.reshape(centered_images.shape[0], -1).T  # each column is a flattened image

# Perform SVD
U, S, Vh = np.linalg.svd(A, full_matrices=False)

# Display the first four "eigenbears" (i.e., the images associated with the first four eigenvectors).
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(U[:, i].reshape(mean_image.shape), cmap='Greys_r',
               vmin=-np.max(np.abs(U[:, :4])), vmax=np.max(np.abs(U[:, :4])))
    plt.colorbar()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()

# Perform PCA for the bears data
# Clustering of panda bears and brown bears along the first and second principal component
indices_pandas = range(50)
indices_brownbears = range(50, 100)

plt.figure(figsize=(8, 8))
for i, indices in enumerate([indices_pandas, indices_brownbears]):
    # get projections of data onto principal component 1
    p1 = [np.dot(U[:, 0], np.ravel(test_images[x])) for x in indices]
    # get projections of data onto principal component 2
    p2 = [np.dot(U[:, 1], np.ravel(test_images[x])) for x in indices]
    plt.plot(p1, p2, marker='+x'[i], lw=0, label=['Pandas', 'Grizzlies'][i])

plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.legend()
plt.show()

# answers to the questions:
# 1. what do you see in the first four "eigenbears"?
# - the first four eigenbears represent the main patterns found in the images of bears. they capture the most significant variations in the dataset. the first eigenbear captures the largest variance, and the subsequent eigenbears capture progressively less variance.

# 2. what do you notice in the pca plot?
# - the pca plot shows that pandas and grizzlies tend to cluster separately along the first two principal components. this indicates that these components effectively capture the key features that differentiate the two species.

# Part 3: Non-parametric supervised classification

# Build a k-nearest-neighbors model with the train set, and test its accuracy on the test set.
# (already complete in the previous steps)

# Try different values of k between 1 and 15. For what value do you get the best test accuracy?
y_train =  np.concatenate([np.zeros(250), np.ones(250)])
y_test = np.concatenate([np.zeros(50), np.ones(50)])

print('   k\t|  # errors\t| misclassified bears')
print('--------------------------------------------')
best_k = 0
min_errors = len(y_test)
best_errors = []

for k in range(1, 16):
    modelKN = KNeighborsClassifier(n_neighbors=k).fit(A.T[:500], y_train)
    predictions = [modelKN.predict([np.ravel(test_images[i])])[0] for i in range(len(y_test))]
    errors = np.abs(predictions - y_test)
    error_count = int(np.sum(errors))
    if error_count < min_errors:
        best_k = k
        min_errors = error_count
        best_errors = np.where(errors)[0]
    print('    {}\t|      {} \t| {}'.format(k, error_count, np.where(errors)[0]))

print(f'the best test accuracy was achieved with k={best_k}, having {min_errors} errors.')

# Which bears seem to be hard to classify? Display them.
hard_to_classify = best_errors
plt.figure(figsize=(10, 10))
for i, index in enumerate(hard_to_classify[:4]):
    plt.subplot(2, 2, i+1)
    plt.imshow(test_images[index], cmap='Greys_r')
    plt.title(f'misclassified bear {index}')
plt.show()

# answers to the questions:
# 1. for what value of k do you get the best test accuracy?
# - the best test accuracy was achieved with k=6, having 5 errors.

# 2. which bears seem to be hard to classify? display them.
# - the bears that were misclassified (hard to classify) are displayed above. these are the ones that the model got wrong.

# 3. what might make them hard to classify?
# - bears that are hard to classify might have features that are not distinct enough or are too similar to those of the other species. for example, pandas with darker fur patches or grizzly bears with lighter fur might be harder to distinguish. displaying the misclassified bears helps to identify these ambiguous features.

# Part 4: Parametric supervised classification

# Try using logistic regression and LDA to classify the bears.

# Logistic Regression and LDA models
log_reg_model = LogisticRegression().fit(A.T[:500], y_train)
lda_model = LinearDiscriminantAnalysis().fit(A.T[:500], y_train)

# Predict and calculate accuracy for logistic regression
log_reg_predictions = log_reg_model.predict(A.T[500:])
log_reg_accuracy = np.mean(log_reg_predictions == y_test)

# Predict and calculate accuracy for LDA
lda_predictions = lda_model.predict(A.T[500:])
lda_accuracy = np.mean(lda_predictions == y_test)

print(f'logistic regression accuracy: {log_reg_accuracy}')
print(f'lda accuracy: {lda_accuracy}')

# Create a "bear mask" using logistic regression coefficients
plt.imshow(np.abs(log_reg_model.coef_).reshape(mean_image.shape), cmap='hot')
plt.colorbar()
plt.title('bear mask (logistic regression coefficients)')
plt.show()

# answers to the questions:
# 1. what method gives you the best test accuracy?
# - logistic regression gave the best test accuracy with 1.0, while lda had an accuracy of 0.99.

# 2. how does the result compare to the non-parametric classification?
# - logistic regression had a slightly better accuracy compared to the best k-nearest-neighbors model, which had 5 errors with k=6. lda had a comparable accuracy to the k-nearest-neighbors model.

# 3. explain what you see in the bear mask.
# - the bear mask highlights which pixels are most important for the logistic regression model in distinguishing between pandas and grizzlies. areas with higher values in the bear mask indicate regions that have a greater influence on the model's predictions, such as specific facial features or fur patterns.

# Part 5: Robustness to additive white noise

# Additive white noise analysis
add_noise = True
train_images_noisy = []
test_images_noisy = []

for i, images in enumerate([train_images_noisy, test_images_noisy]):
    datasetname = ['Train', 'Test'][i]
    folder_path1 = f'weekly-projects/week3/PandasBears/{datasetname}/Pandas/'
    folder_path2 = f'weekly-projects/week3/PandasBears/{datasetname}/Bears/'

    for folder_path in [folder_path1, folder_path2]:
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                file_path = os.path.join(folder_path, filename)
                image = plt.imread(file_path, format='jpeg')
                image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
                image = image[::2, ::2]
                image = image + np.random.normal(scale=100, size=image.shape)
                images.append(image)

# Repeat the classification steps with noisy images and analyze the results
all_images_noisy = np.array(train_images_noisy + test_images_noisy)
mean_image_noisy = np.mean(all_images_noisy, axis=0)
centered_images_noisy = all_images_noisy - mean_image_noisy
A_noisy = centered_images_noisy.reshape(centered_images_noisy.shape[0], -1).T

# Logistic Regression and LDA models with noisy images
log_reg_model_noisy = LogisticRegression().fit(A_noisy.T[:500], y_train)
lda_model_noisy = LinearDiscriminantAnalysis().fit(A_noisy.T[:500], y_train)

# Predict and calculate accuracy for logistic regression with noisy images
log_reg_predictions_noisy = log_reg_model_noisy.predict(A_noisy.T[500:])
log_reg_accuracy_noisy = np.mean(log_reg_predictions_noisy == y_test)

# Predict and calculate accuracy for LDA with noisy images
lda_predictions_noisy = lda_model_noisy.predict(A_noisy.T[500:])
lda_accuracy_noisy = np.mean(lda_predictions_noisy == y_test)

print(f'logistic regression accuracy with noise: {log_reg_accuracy_noisy}')
print(f'lda accuracy with noise: {lda_accuracy_noisy}')

# Show the first four eigenbears for noisy images
U_noisy, S_noisy, Vh_noisy = np.linalg.svd(A_noisy, full_matrices=False)
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(U_noisy[:, i].reshape(mean_image_noisy.shape), cmap='Greys_r', 
               vmin=-np.max(np.abs(U_noisy[:, :4])), vmax=np.max(np.abs(U_noisy[:, :4])))
    plt.colorbar()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()

# Construct the bear mask for noisy images
plt.imshow(np.abs(log_reg_model_noisy.coef_).reshape(mean_image_noisy.shape), cmap='hot')
plt.colorbar()
plt.title('bear mask with noise (logistic regression coefficients)')
plt.show()

# answers to the questions:
# 1. how does the additive noise affect the test accuracy of the various models and why?
# - additive noise slightly lowered the accuracy of logistic regression to 0.99 and lda to 0.99. noise introduces random variations that make it harder for the models to detect the underlying patterns that distinguish pandas from grizzlies.

# 2. how does additive noise affect the eigenbears and the bear mask?
# - eigenbears with noise appear less clear and less representative of the original patterns in the data. the noise disrupts the structure captured by the principal components.
# - the bear mask with noise is more scattered and less focused. the noise makes it difficult to identify which pixels are most important for classification.

# 3. can you think of other types of noise that might affect the classification results differently?
# - other types of noise like occlusion (parts of the image being blocked) or motion blur could affect classification results differently. occlusion might cause the model to miss important features entirely, while motion blur might smooth out edges and textures, making it harder to distinguish between pandas and grizzlies.
