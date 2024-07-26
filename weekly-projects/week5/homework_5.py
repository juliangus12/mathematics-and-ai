# Homework 5 (due 07/30/2024)

# standard imports
import os, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier
from PIL import Image

# Part 1: Binary classification of synthetic data

# Part 1.1: Generate and explore synthetic data
# The next cell defines the function generate_dataset, which you can use to generate synthetic (i.e., computer generated) data sets for binary classification. It includes eight different methods for data-set generation.
# Try out each method and visualize the resulting data set. For the 'swiss' and 'scurve' data sets, try out two different values of the keyword argument `splits`.
# Comment on WHETHER and WHY you anticipate this data set to be relatively easy or relatively hard to classify with a linear classifier.
# Comment on WHETHER and WHY you anticipate this data set to be relatively easy or relatively hard to classify with a nonlinear classifier.

# function to convert an array of real numbers into an array of 0s and 1s
def binarize(arr, split=10):
    percentiles = int(np.ceil(100 / split))
    split_points = np.arange(0, 100 + percentiles, percentiles)
    split_points[split_points > 100] = 100
    deciles = np.percentile(arr, split_points)
    modified_arr = np.zeros_like(arr)
    for i in range(split):
        if i == split - 1:
            if i % 2 == 0:
                modified_arr[(arr >= deciles[i])] = 0
            else:
                modified_arr[(arr >= deciles[i])] = 1
        else:
            if i % 2 == 0:
                modified_arr[(arr >= deciles[i]) & (arr < deciles[i + 1])] = 0
            else:
                modified_arr[(arr >= deciles[i]) & (arr < deciles[i + 1])] = 1
    return modified_arr

# function to generate datasets
def generate_dataset(dataset_type, n_samples=300, noise=0.1, split=10, random_state=0):
    if dataset_type == 'linearly_separable':
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
                                   random_state=random_state, n_clusters_per_class=1)
    elif dataset_type == 'blobs':
        X, y = make_blobs(n_samples=[n_samples // 2, n_samples // 2], random_state=random_state, cluster_std=noise)
    elif dataset_type == 'quantiles':
        X, y = make_gaussian_quantiles(n_samples=n_samples, n_classes=2, cov=noise, random_state=random_state)
    elif dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    elif dataset_type == 'unstructured':
        X, y = np.random.random(size=(n_samples, 2)), np.random.randint(0, 2, size=(n_samples))
    elif dataset_type == 'swiss':
        X, y = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
        X = np.array([X[:, 0], X[:, 2]]).T
        y = binarize(y, split=split)
    elif dataset_type == 'scurve':
        X, y = make_s_curve(n_samples=n_samples, noise=noise, random_state=random_state)
        X = np.array([X[:, 0], X[:, 2]]).T
        y = binarize(y, split=split)
    else:
        raise ValueError("Invalid dataset type")
    X = StandardScaler().fit_transform(X)
    return X, y

# function to visualize datasets
def visualize_dataset(dataset_type, split_values=[10]):
    for split in split_values:
        X, y = generate_dataset(dataset_type, split=split)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        plt.title(f"{dataset_type} with split={split}")
        plt.show()

# visualize datasets
datasets = ['linearly_separable', 'blobs', 'quantiles', 'moons', 'circles', 'unstructured', 'swiss', 'scurve']
for dataset in datasets:
    if dataset in ['swiss', 'scurve']:
        visualize_dataset(dataset, split_values=[2, 10])
    else:
        visualize_dataset(dataset)

# comments on anticipated difficulty of classification
# linearly_separable: this dataset will be relatively easy to classify with a linear classifier because the data points are linearly separable. each class forms a distinct cluster that can be divided by a straight line. nonlinear classifiers will also perform well due to the simplicity of the data structure.
# blobs: this dataset will be easy for both linear and nonlinear classifiers because the data points form distinct, well-separated clusters. the clusters have clear boundaries that can be captured by both types of classifiers.
# quantiles: this dataset will be hard for a linear classifier because the data points are not linearly separable. the boundaries between classes are curved, making it challenging for a linear model to perform well. however, a nonlinear classifier should handle it well because it can capture the curved boundaries.
# moons: similar to the quantiles dataset, the moons dataset will be challenging for linear classifiers due to its curved boundaries between classes. nonlinear classifiers will perform better as they can adapt to the curved decision boundaries required to separate the classes.
# circles: this dataset is particularly hard for linear classifiers because the classes form concentric circles. a linear decision boundary cannot separate these classes effectively. nonlinear classifiers, like those using radial basis functions, will handle this dataset well by capturing the circular structure.
# unstructured: this dataset will be hard for both linear and nonlinear classifiers because the data points are randomly distributed without any clear pattern or separable structure. classifying such data will be challenging for any model.
# swiss: the swiss roll dataset will be challenging for linear classifiers due to its complex, nonlinear structure. nonlinear classifiers will perform better, especially with appropriate kernel functions that can capture the manifold structure.
# scurve: similar to the swiss roll, the scurve dataset is hard for linear classifiers due to its 3d-like curved structure. nonlinear classifiers will perform better by adapting to the complex boundaries.

# Part 1.2: SVM with nonlinear kernels
# The kernel comparison currently produces only visual results. Add code to the function so that it also outputs train and test accuracy of the different SVMs. (Note: Think carefully about where the right place in the code is to do a train-test split.)
# Run the kernel comparison for the data sets from Part 1.1. Do the results confirm or contradict your expectations that you formulated in Part 1.1.? Did any of the results surprise you?
# Consult sklearn's documentation to learn how the keyword arguments `degree` and `gamma` affect your classifier. Try out a few different values of these parameters. How and what can one infer from the shape of the decision boundary about the classifier's `degree` or `gamma`?

# function to compare kernels
def kernel_comparison(X, y, support_vectors=True, tight_box=False, if_flag=False):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    fig = plt.figure(figsize=(10, 3))
    for ikernel, kernel in enumerate(['linear', 'poly', 'rbf', 'sigmoid']):
        clf = SVC(kernel=kernel, degree=3, gamma='scale').fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        results[kernel] = (train_acc, test_acc)
        ax = plt.subplot(1, 4, 1 + ikernel)
        common_params = {"estimator": clf, "X": X, "ax": ax}
        DecisionBoundaryDisplay.from_estimator(
            **common_params,
            response_method="predict",
            plot_method="pcolormesh",
            alpha=0.3,
        )
        DecisionBoundaryDisplay.from_estimator(
            **common_params,
            response_method="decision_function",
            plot_method="contour",
            levels=[-1, 0, 1],
            colors=["k", "k", "k"],
            linestyles=["--", "-", "--"],
        )
        if support_vectors:
            ax.scatter(
                clf.support_vectors_[:, 0],
                clf.support_vectors_[:, 1],
                s=150,
                facecolors="none",
                edgecolors="k",
            )
        ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
        ax.set_title(f"{kernel}\nTrain Acc: {train_acc:.2f}\nTest Acc: {test_acc:.2f}")
        ax.axis('off')
        if tight_box:
            ax.set_xlim([X[:, 0].min(), X[:, 0].max()])
            ax.set_ylim([X[:, 1].min(), X[:, 1].max()])
    plt.show()
    return results

# show results of kernel comparison for datasets from part 1
for dataset in datasets:
    X, y = generate_dataset(dataset)
    results = kernel_comparison(X, y)
    print(f"Results for {dataset}: {results}")

# to summarize the results of the kernel comparison, the results generally confirm the expectations. linear kernel worked well for linearly separable data while rbf and poly kernels worked better for complex datasets. the unstructured dataset remained hard to classify for all kernels.

# examine effect of degree and gamma keyword
def experiment_with_params(X, y, degree=3, gamma='scale'):
    clf = SVC(kernel='poly', degree=degree, gamma=gamma).fit(X, y)
    DecisionBoundaryDisplay.from_estimator(clf, X, plot_method="pcolormesh", alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
    plt.title(f"Degree: {degree}, Gamma: {gamma}")
    plt.show()

X, y = generate_dataset('moons')
experiment_with_params(X, y, degree=5, gamma=0.1)

# the degree argument affects the poly kernel. it changes the model by making the decision boundary more complex and flexible to fit the data. this affects the model's bias-variance tradeoff by increasing variance and decreasing bias. as one increases the degree, the decision boundary becomes more intricate and can fit more complex patterns in the data.
# the gamma argument affects the rbf kernel. it changes the model by determining the influence of a single training example. this affects the model's bias-variance tradeoff by increasing variance and decreasing bias. as one increases gamma, the decision boundary becomes more sensitive to individual data points, making it more wiggly and complex.
# from the shape of the decision boundary, one can infer the degree or gamma value. a higher degree results in more complex, intricate decision boundaries for the poly kernel, while a higher gamma makes the decision boundary more sensitive and wiggly for the rbf kernel.

# Part 2: US Flags
# Part 2.1: Load and explore flags data
# The function load_images loads the image data from the flags folder and turns each image into a binary (i.e., black and white) array.
# Load the flags data.
# Display four flags of your choice in a figure. Use the matplotlib commands subplot and imshow to create a figure with 2x2 flags. Consult the matplotlib documentation to find a way set the aspect ratio of your displayed flags to match their original aspect ratio. Update your code accordingly.

def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')  # Convert image to black and white
            img = np.array(img) // (256 // 2)  # Convert to BW
            images.append(img)
            labels.append(filename.split('.')[0])  # Extract the state code as label
    return images, labels

folder_path = '/home/julian/mathematics-and-ai/flags'  
images, labels = load_images(folder_path)

# display four black-and-white flags in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, img, label in zip(axes.flatten(), images[:4], labels[:4]):
    ax.imshow(img, cmap='gray', aspect='auto')
    ax.set_title(label)
    ax.axis('off')
plt.show()

# Part 2.2: SVMs for flag pixel data
# The function sample_pixels samples a pixel from a given image uniformly at random.
# Use the sample_pixels function to generate synthetic data sets of pixels from for a flag image.
# Update the kernel_comparison function so that if if_flag is True the decision boundaries are plotted in a 2x2 grid of subplots with plot ranges matching the height and width of the flags.
# Show the results of the kernel comparison for the four flags that your previously selected. Use the highest values of degree and gamma that still run reasonably fast on your laptop.
# Adjust your code so that you can run the quantitative part (i.e., the calculation of train and test accuracy) without plotting the decision boundaries. Run the adjusted code on all flags to identify for each kernel the flags that yield to best easiest-to-classify and hardest-to-classify data sets. Test how the number of pixels sampled affects your results.

def sample_pixels(image, num_samples=100):
    height, width = image.shape
    pixel_data = np.zeros((num_samples, 2))
    pixel_labels = np.zeros(num_samples)
    for i in range(num_samples):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        pixel_data[i, :] = [x / width, y / height]
        pixel_labels[i] = image[y, x]
    return pixel_data, pixel_labels

# run kernel comparison for selected flags
for image, label in zip(images[:4], labels[:4]):
    pixels, labels = sample_pixels(image, num_samples=1000)
    results = kernel_comparison(pixels, labels, if_flag=True)
    print(f"Results for {label}: {results}")

# non-visual kernel comparison for all flags
all_results = {}
for image, label in zip(images, labels):
    pixels, labels = sample_pixels(image, num_samples=1000)
    results = kernel_comparison(pixels, labels, support_vectors=False, tight_box=False, if_flag=False)
    all_results[label] = results

# print the results
for label, results in all_results.items():
    print(f"Results for {label}: {results}")

# summarize the results for linear kernel
linear_best = sorted(all_results.items(), key=lambda x: x[1]['linear'][1], reverse=True)[:3]
linear_worst = sorted(all_results.items(), key=lambda x: x[1]['linear'][1])[:3]
print("Linear kernel performed best on the flags of:", [x[0] for x in linear_best])
print("Linear kernel performed worst on the flags of:", [x[0] for x in linear_worst])

# summarize the results for polynomial kernel
poly_best = sorted(all_results.items(), key=lambda x: x[1]['poly'][1], reverse=True)[:3]
poly_worst = sorted(all_results.items(), key=lambda x: x[1]['poly'][1])[:3]
print("Polynomial kernel performed best on the flags of:", [x[0] for x in poly_best])
print("Polynomial kernel performed worst on the flags of:", [x[0] for x in poly_worst])

# summarize the results for rbf kernel
rbf_best = sorted(all_results.items(), key=lambda x: x[1]['rbf'][1], reverse=True)[:3]
rbf_worst = sorted(all_results.items(), key=lambda x: x[1]['rbf'][1])[:3]
print("RBF kernel performed best on the flags of:", [x[0] for x in rbf_best])
print("RBF kernel performed worst on the flags of:", [x[0] for x in rbf_worst])

# summarize the results for sigmoid kernel
sigmoid_best = sorted(all_results.items(), key=lambda x: x[1]['sigmoid'][1], reverse=True)[:3]
sigmoid_worst = sorted(all_results.items(), key=lambda x: x[1]['sigmoid'][1])[:3]
print("Sigmoid kernel performed best on the flags of:", [x[0] for x in sigmoid_best])
print("Sigmoid kernel performed worst on the flags of:", [x[0] for x in sigmoid_worst])

# Part 2.3: Comparison to decision trees
# Decision trees and SVMs yield substantially different decision boundaries.
# An arbitrarily complex decision tree would be able to achieve perfect training accuracy on any data set. Explain why.
# For a very large data set of flag pixels, an arbitrarily complex decision tree is likely to achieve (almost) perfect test accuracy as well. Explain why.
# Select four flags for which you anticipate a simple decision tree to outperform all your SVMs. Write code that fits a decision tree to a flag pixel data set. Use your code to check your hypothesis.

# an arbitrarily complex decision tree would be able to achieve perfect training accuracy on any data set, because it can memorize the data exactly by creating a unique path for each training example.
# for a very large data set of flag pixels, an arbitrarily complex decision tree is likely to achieve almost perfect test accuracy because the large dataset provides enough information for the tree to generalize well and make accurate predictions on unseen data.
# a simple decision tree is likely to perform well on the sampled pixel data of the flags of: montana (mt), north dakota (nd), new york (ny), virginia (va)

def compare_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier().fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"Decision Tree Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
    DecisionBoundaryDisplay.from_estimator(clf, X, plot_method="pcolormesh", alpha=0.3)
    plt.title("Decision Tree")
    plt.show()

# comparison of svm and decision tree performance on sampled pixel data for four flags
for image, label in zip(images[:4], labels[:4]):
    pixels, labels = sample_pixels(image, num_samples=1000)
    print(f"Results for {label}:")
    compare_decision_tree(pixels, labels)
