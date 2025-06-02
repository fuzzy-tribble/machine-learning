import platform
import psutil

try:
    import GPUtil
    gpu_available = True
except ImportError:
    gpu_available = False

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML


def plot_cv_indices(cv, X, y, ax, n_splits, lw=10):
    # https://www.geeksforgeeks.org/cross-validation-using-k-fold-with-scikit-learn/
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes at the end
    ax.scatter(range(len(X)), [n_splits + 0.5] * len(X), c=y, marker="_", lw=lw)

    # Formatting
    yticklabels = list(range(n_splits)) + ["class"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 1.2, -0.2],
        xlim=[0, len(X)],
    )
    ax.set_title(f"{type(cv).__name__} (k={n_splits})", fontsize=15)
    return ax

def plot_feature_statistics(dataframe, feature_names, line=True, hist=True, box=True):
    columns = [line, hist, box].count(True)
    fig, axes = plt.subplots(len(feature_names), columns, figsize=(4 * columns, 1.5 * len(feature_names)))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # handle if only one type of plot is selected
    if columns == 1:
        axes = axes.reshape(-1, 1)

    for i, feature in enumerate(feature_names):
        col_idx = 0
        if line:
            sns.lineplot(data=dataframe[feature], ax=axes[i, col_idx])
            axes[i, col_idx].set_title(f'Lineplot of {feature}')
            col_idx += 1
        
        if hist:
            sns.histplot(data=dataframe[feature], kde=True, ax=axes[i, col_idx])
            axes[i, col_idx].set_title(f'Histogram of {feature}')
            col_idx += 1
        
        if box:
            # Boxplot
            # whiskers represent the range (min/max) of the data (excluding outliers)
            # points outside the whiskers are considered outliers
            # box represents the interquartile range (IQR): middle 50% of the data
            # read eg. if the median is centered in the box and the whiskers are of equal length, the data is symmetric => symmetric/normal distribution
            sns.boxplot(data=dataframe[feature],orient='h', ax=axes[i, col_idx])
            axes[i, col_idx].set_title(f'Boxplot of {feature}')
            col_idx += 1
    
    plt.tight_layout()
    return fig, axes

def disp_df(df, max_height=300, max_width=600):
    style = f'''
    <style>
    .scrollable_df {{
        max-height: {max_height}px;
        max-width: {max_width}px;
        overflow: auto;
        display: inline-block;
    }}
    </style>
    '''
    display(HTML(style + df.to_html(classes='scrollable_df')))

def disp_hw():
    """
    Display hardware information
    """
    # CPU information
    print("CPU Info:")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    print(f"System: {platform.system()} {platform.version()}")
    
    # Memory information
    print("\nMemory Info:")
    mem = psutil.virtual_memory()
    print(f"Total: {mem.total / (1024 ** 3):.2f} GB")
    print(f"Available: {mem.available / (1024 ** 3):.2f} GB")
    print(f"Used: {mem.used / (1024 ** 3):.2f} GB")
    print(f"Percentage: {mem.percent}%")
    
    # GPU information
    if gpu_available:
        print("\nGPU Info:")
        gpus = GPUtil.getGPUs()
        if gpus:
            for gpu in gpus:
                print(f"GPU: {gpu.name}")
                print(f"Total Memory: {gpu.memoryTotal / 1024:.2f} GB")
                print(f"Free Memory: {gpu.memoryFree / 1024:.2f} GB")
                print(f"Used Memory: {gpu.memoryUsed / 1024:.2f} GB")
                print(f"GPU Load: {gpu.load * 100:.2f}%")
        else:
            print("No GPU found")
    else:
        print("\nGPUtil is not installed. GPU information will not be displayed.")


def plot_validation_curve(x_train, y_train, train_scores, val_scores, param_range, **kwargs):
    # plot validation curve for max_depth (and get optimal value and its score)
    plt.figure()

    # mean and standard deviation over cross-validation folds
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # Filling in between mean + std and mean - std
    x_axis = np.arange(1, 11)  # param range
    train_label = "Training"
    val_label = "Cross-Validation"

    train_plot = plt.plot(x_axis, train_scores_mean, "-o", markersize=2, label=train_label)
    val_plot = plt.plot(x_axis, val_scores_mean, "-o", markersize=2, label=val_label)

    plt.fill_between(
        x_axis,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color=train_plot[0].get_color(),
    )
    plt.fill_between(
        x_axis,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.1,
        color=val_plot[0].get_color(),
    )

    #  Find the parameter value that maximize the mean of the validation scores over cross-validation folds
    best_max_depth = param_range[np.argmax(val_scores_mean)]
    score = val_scores_mean[np.argmax(val_scores_mean)]
    print("--> max_depth = {} --> score = {:.4f}".format(best_max_depth, score))

    plt.title(
        "DT Validation Curve on Wine Dataset ({nfolds}-Fold Cross Validation)".format(
            nfolds=nfold
        )
    )
    plt.legend(loc="best")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.xscale("linear")
    plt.ylim(0.5, 1.01)

    plt.savefig("{}_max_depth-{nfolds}-fold.png".format(model["name"], nfolds=nfold))

    plt.show()


## LEARNING CURVE DISPLAY CUSTOM PLOT
# # customize plot
# plt.figure()

# train_score_mean = train_scores.mean(axis=1)
# train_score_std = train_scores.std(axis=1)
# test_score_mean = test_scores.mean(axis=1)
# test_score_std = test_scores.std(axis=1)

# plt.plot(
#     lcd.train_sizes,
#     train_score_mean,
#     marker="o",
#     markersize=2,
#     color="blue",
#     label="Training",
# )
# plt.fill_between(
#     lcd.train_sizes,
#     train_score_mean - train_score_std,
#     train_score_mean + train_score_std,
#     alpha=0.1,
#     color="blue",
# )

# plt.plot(
#     lcd.train_sizes,
#     lcd.test_scores.mean(axis=1),
#     marker="o",
#     markersize=2,
#     color="red",
#     label="Cross-Validation",
# )
# plt.fill_between(
#     lcd.train_sizes,
#     test_score_mean - test_score_std,
#     test_score_mean + test_score_std,
#     alpha=0.1,
#     color="red",
# )

# plt.title("DT Learning Curves \nX-Fold Cross Validation")  # Customize title
# plt.xlabel(lcd.ax_.xaxis.label.get_text())  # Customize x-axis label
# plt.ylabel("Accuracy")  # Customize y-axis label
# plt.legend(loc="best")  # Customize legend location

# RES_DIR = "results/"
# model_name = "dt"
# plt.savefig(RES_DIR + "{}_learning_curve".format(model_name))