import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def processDataTest(df: pd.DataFrame,
                    
                        ) -> pd.DataFrame:
    


def testA1N09(model: RandomForestRegressor):
    df = pd.read_csv("testA1N09.csv")
    df = processDataTest(df)


# if __name__=="__main__":
#     rf = RandomForestRegressor()
#     testA1N09(rf)
