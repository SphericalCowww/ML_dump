import sys, math
import numpy as np
import pandas as pd
from matplotlib import style
style.use("ggplot")

if __name__ == "__main__":
    print("---------------------------------------------------------------------");
    data = {"a":[1, 2, 3, 4],
            "b":[221082, 423, 493, 32149],
            "c":["d", "s", "ad", "ds"]};
    df = pd.DataFrame(data);
    df.set_index("a", inplace=True);
    print(df[["c", "b"]]);
    print();
    print(np.array(df[["c", "b"]]));






