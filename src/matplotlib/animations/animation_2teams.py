import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate some random data for the first team
n = 100
x_team1 = np.random.uniform(0, 40, n)
y_team1 = np.random.uniform(0, 20, n)

# Generate some random data for the second team
x_team2 = np.random.uniform(0, 40, n)
y_team2 = np.random.uniform(20, 40, n)

# Create a pandas dataframe for each team
df_team1 = pd.DataFrame({"x": x_team1, "y": y_team1})
df_team2 = pd.DataFrame({"x": x_team2, "y": y_team2})

# Create a figure and axis
fig, ax = plt.subplots()

# Set the x and y limits of the plot
ax.set_xlim(0, 40)
ax.set_ylim(0, 40)

# Create empty scatter plots for each team
scatter_team1 = ax.scatter([], [], label="Team 1")
scatter_team2 = ax.scatter([], [], label="Team 2")

# Add a legend to the plot
ax.legend()

# Define the initialization function
def init():
    # scatter_team1.set_offsets([])
    # scatter_team2.set_offsets([])
    return (
        scatter_team1,
        scatter_team2,
    )


# Define the animation function
def animate(i):
    # Get the x and y values for each team from the dataframes
    x_team1 = df_team1.iloc[i]["x"]
    y_team1 = df_team1.iloc[i]["y"]
    x_team2 = df_team2.iloc[i]["x"]
    y_team2 = df_team2.iloc[i]["y"]

    # Update the scatter plots with the new values
    scatter_team1.set_offsets([[x_team1, y_team1]])
    scatter_team2.set_offsets([[x_team2, y_team2]])

    return (
        scatter_team1,
        scatter_team2,
    )


# Create the animation
animation = FuncAnimation(
    fig, animate, init_func=init, frames=n, interval=50, blit=True
)

# Display the animation
# from IPython.display import HTML
# HTML(animation.to_jshtml())
