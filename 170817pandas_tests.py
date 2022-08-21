import sys, math
import numpy as np
import pandas as pd
from matplotlib import style
style.use("ggplot")

if __name__ == "__main__":
    df1 = pd.DataFrame({"a":[0, 4, 5, 8],
                        "b":[0, 0, 1, 0],
                        "c":[123, 213, 435, 382]},
                        index = [1, 2, 3, 4]);
    df2 = pd.DataFrame({"a":[0, 2, 5, 6],
                        "k":[0, 0, 1, 1],
                        "d":[943, 109, 349, 219]},
                        index = [1, 2, 6, 7]);
    #df3 = df1.append(df2);
    #df3 = pd.concat([df1, df2]);
    #df3 = pd.merge(df1, df2, on="a", how="right");
    df1.set_index("a", inplace=True);
    df2.set_index("a", inplace=True);
    df3 = df1.join(df2);
    print(df3);





