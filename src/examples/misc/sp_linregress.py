import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


df = pd.read_csv("../../data/HousePrices.csv")

print(df.head(6))

x = df["Area (Sq. ft)"]
y = df["Price (£)"]
plt.scatter(x, y, marker="x")
plt.title("House Price vs. Area (Sq. ft)", fontsize=20)
plt.xlabel(list(df)[0], fontsize=16)
plt.ylabel(list(df)[2], fontsize=16)
plt.tight_layout()
plt.show()

gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print("\nGradient and intercept", gradient, intercept)
print("R-squared (correlation coefficient)", r_value ** 2)
print("p-value", p_value)
print("Standard error:", std_err)

plt.plot(x, y, "x", label="Data")
plt.plot(x, gradient * x + intercept, color="r", label="fitted line")
plt.title("House Price vs. Area (Sq. ft)", fontsize=20)
plt.xlabel(list(df)[0], fontsize=16)
plt.ylabel(list(df)[2], fontsize=16)
plt.legend(loc="best")
plt.tight_layout()
plt.show()
