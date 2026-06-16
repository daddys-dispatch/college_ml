import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.datasets import fetch_california_housing

housing_df = fetch_california_housing(as_frame=True).frame
numerical_features = housing_df.select_dtypes(include=[np.number]).columns


def plot(plot_type, title_prefix, **kwargs):
    plt.figure(figsize=(15, 10))

    for i, feature in enumerate(numerical_features, start=1):
        plt.subplot(3, 3, i)
        plot_type(housing_df, x=feature, **kwargs)
        plt.title(f"{title_prefix} {feature}")

    plt.tight_layout()
    plt.show()


print(housing_df.describe())
plot(sns.histplot, "Distribution of", kde=True, bins=30)

for feature in numerical_features:
    series = housing_df[feature]

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    print(f"{feature}: {outliers} outliers")

plot(sns.boxplot, "Box Plot of")
