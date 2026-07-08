import sys
import matplotlib.pyplot as plt

def plot_metrics(epochs, *args, titles=None, labels=None, **kwargs):
    """
    Plot metrics dynamically side-by-side.
    
    Args:
        epochs: Array of epoch numbers
        *args: Positional arguments for metrics (e.g., train_loss, train_acc)
        titles: List of titles for each metric plot
        labels: List of labels for each metric group
        **kwargs: Keyword arguments for metrics (e.g., Loss=[train_loss, val_loss], Accuracy=[train_acc, val_acc])
    Example:
        plot_metrics(epochs, train_loss, train_acc, titles=['Loss', 'Accuracy'])
        plot_metrics(epochs, Loss=[train_loss, val_loss], Accuracy=[train_acc, val_acc], labels=[('Train loss', 'Val loss'), ('Train acc', 'Val acc')])
    """
    plt.close('all')
    
    # 1. Route based on how user passed the data
    is_keyword_mode = len(kwargs) > 0
    metric_source = kwargs if is_keyword_mode else {f"Metric {i+1}": data for i, data in enumerate(args)}
    
    num_plots = len(metric_source)
    if num_plots == 0: 
        return
    
    # 2. Initialize the subplot grid
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 3.5))
    if num_plots == 1: axes = [axes]
    
    # Apply user override titles if positional mode is active
    if not is_keyword_mode and titles and len(titles) >= num_plots:
        display_titles = titles
    else:
        display_titles = list(metric_source.keys())
        
    styles = ['-', '--', ':', '-.']
    
    # 3. Render each metric chart
    for idx, (ax, (key_name, data_input)) in enumerate(zip(axes, metric_source.items())):
        
        # Standardize single lists, tuples, or nested lists into an iterable container
        if not isinstance(data_input, (list, tuple)):
            data_groups = [data_input]
        else:
            # Check if it's a nested list/tuple structure like [train, val]
            if len(data_input) > 0 and isinstance(data_input[0], (list, tuple)):
                data_groups = data_input
            else:
                # It's a flat list/tuple representing a single metric line
                data_groups = [data_input]
                
        # Draw the target lines on the chart panel
        for j, data_line in enumerate(data_groups):
            # if labels is None, remove labels entirely and use custom labels for each individual plot
            if labels is None:
                lbl = None
            elif isinstance(labels[0], (list, tuple)):
                # labels per subplot
                lbl = labels[idx][j] if j < len(labels[idx]) else f'Group {j+1}'
            else:
                # one label list shared by all plots
                lbl = labels[j] if j < len(labels) else f'Group {j+1}'
            ax.plot(epochs, data_line, linestyle=styles[j % 4], linewidth=2, label=lbl)
        ax.set_title(display_titles[idx], fontsize=12, fontweight='bold')
        ax.set_xlabel('Epochs')
        if len(data_groups) > 1 and labels is not None: ax.legend(loc='best')
        elif len(data_groups) == 1 and labels is None: ax.legend(labels=[key_name], loc='best')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()

    # 4. Environment-aware rendering engine
    if 'ipykernel' in sys.modules:
        from IPython.display import clear_output, display
        clear_output(wait=True)
        display(fig)
    else:
        plt.ion()
        fig.canvas.draw()
        fig.canvas.flush_events()
