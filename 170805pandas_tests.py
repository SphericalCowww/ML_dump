import sys, math
import numpy as np
import pandas as pd
from matplotlib import style
style.use("ggplot")

if __name__ == "__main__":
    print("---------------------------------------------------------------------");
    df = pd.read_csv("breast-cancer-wisconsin.data", index_col = 0);
    print(df.head());
    #df.to_csv("breast-cancer-wisconsin.csv", header=False);





