import pandas as pd


def find_s(file):
    df = pd.read_csv(file)
    h = ["?"] * (len(df.columns) - 1)

    for _, row in df.iterrows():
        if row.iloc[-1] == "Yes":
            for i, v in enumerate(row[:-1]):
                h[i] = v if h[i] in ("?", v) else "?"

    print(f"{df}\n\nFinal Hypothesis: {h}")


find_s("../weather_data.csv")
