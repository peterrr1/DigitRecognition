import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Function to create a barplot for the number of samples per label in the dataset
def create_barplot(dataset):

    # Get the unique labels and their counts from the dataset
    data = np.unique(dataset.label, return_counts=True)

    x = data[0]  # labels
    y = data[1]  # counts

    # Create a barplot with seaborn
    ax = sns.barplot(x=x, y=y, palette='viridis')

    # Set the labels and title of the plot
    ax.set_ylabel('Number of samples')
    ax.set_xlabel('Label')
    ax.set_title('Number of samples per label')

    # Draw a horizontal line at y=0
    ax.axhline(0, color="k", clip_on=False)

    # Remove the top and right spines from plot
    sns.despine(bottom=True)

    # Save the figure to a file
    ax.get_figure().savefig(f'./plots/{ax.get_title()}.png')

    # Display the plot
    plt.show()


# Function to display a grid of sample images from the dataset
def show_images(dataset):

    # Create a new figure
    figure = plt.figure(figsize=(10, 10))

    rows, cols = 3, 3  # grid size

    # Loop over the grid cells
    for i in range(1, rows * cols + 1):

        # Randomly select an index from the dataset
        sample_idx = torch.randint(0, len(dataset), size = (1,)).item()

        # Get the image and label at the selected index
        img, label = dataset[sample_idx]

        # Add a subplot to the figure
        figure.add_subplot(rows, cols, i)

        # Set the title of the subplot to the label of the image
        plt.title(label.item())

        # Remove the axes from the subplot
        plt.axis('off')

        # Display the image in the subplot
        plt.imshow(img.squeeze(), cmap='gray')

    # Save the figure to a file
    figure.savefig(f'./plots/sample_images.png')

    # Display the figure
    plt.show()

# Function to create a confusion matrix
def create_confusion_matrix(y_true, y_pred, labels):

    # Compute the confusion matrix
    cfm = confusion_matrix(y_true, y_pred, labels = labels)

    # Display the confusion matrix
    ConfusionMatrixDisplay(cfm).plot()

    # Save the figure to a file
    plt.savefig(f'./plots/confusion_matrix.png')
    
    # Display the figure
    plt.show()

