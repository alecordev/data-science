import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate some random data
n = 100
x = np.random.rand(n)
y = np.random.rand(n)

# Create a pandas dataframe from the data
df = pd.DataFrame({"x": x, "y": y})

# Create a figure and axis
fig, ax = plt.subplots()

# Set the x and y limits of the plot
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Create an empty scatter plot
scatter = ax.scatter([], [])

# Define the initialization function
def init():
    # scatter.set_offsets([])
    return (scatter,)


# Define the animation function
def animate(i):
    # Get the x and y values from the dataframe
    x = df.iloc[i]["x"]
    y = df.iloc[i]["y"]

    # Update the scatter plot with the new values
    scatter.set_offsets([[x, y]])
    return (scatter,)


# Create the animation
animation = FuncAnimation(
    fig, animate, init_func=init, frames=len(df), interval=50, blit=True
)

# Display the animation
# from IPython.display import HTML
# HTML(animation.to_jshtml())
