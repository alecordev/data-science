import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the CSV file into a pandas dataframe
df = pd.read_csv("data.csv")

# Create a figure and axis
fig, ax = plt.subplots()

# Set the x and y limits of the plot
ax.set_xlim(df["x"].min(), df["x"].max())
ax.set_ylim(df["y"].min(), df["y"].max())

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
    fig, animate, init_func=init, frames=len(df), interval=200, blit=True
)

# Save the animation as a GIF
animation.save("points.gif", writer="imagemagick")

# Save the animation as an MP4 file
animation.save("points.mp4", writer="ffmpeg")

# Display the animation in a Jupyter Notebook
# from IPython.display import HTML
# HTML(animation.to_jshtml())
