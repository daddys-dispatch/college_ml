import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing

housing_df = fetch_california_housing(as_frame=True).frame
corr_matrix = housing_df.corr()

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of California Housing Features")
plt.show()

sns.pairplot(housing_df, diag_kind="kde")
plt.suptitle("Pair Plot of California Housing Features", y=1)
plt.show()
