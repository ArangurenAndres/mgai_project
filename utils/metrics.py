import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_loss_curve(csv_path: str, title: str = "Training Loss Over Epochs"):
    """
    Plot training loss over epochs from a CSV file.
    
    Args:
        csv_path (str): Path to the loss_over_epochs.csv file.
        title (str): Optional title for the plot.
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Set plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))

    # Plot
    sns.lineplot(data=df, x="epoch", y="loss", marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.show()
