from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
import argparse
import joblib
import numpy as np
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

ds = TabularDatasetFactory.from_delimited_files(path="https://s3.amazonaws.com/static.akk.io/Akkio-NCAAHistoricalData.csv")
pdata = ds.to_pandas_dataframe()
pdata = pdata[pdata["Year"]>2009]
topTenFeatures = ["G1", "G2", "Seed1", "Seed2", "PF1", "PF2", "3P%1", "3P%2", "TOV1", "TOV2", "winner"]
x = pdata[topTenFeatures]
y = x.pop("winner").apply(lambda s: 1 if s == "Team2" else 0)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument("--C", type=float, default=2.0, help="inverse of regularization strength")
    parser.add_argument("--max_iter", type=int, default=500, help="max iterations for convergence")
    parser.add_argument("--penalty", type=str, default="l2", help="norm used in penalization")
    parser.add_argument("--solver", type=str, default="saga", help="algorithm name")

    args = parser.parse_args()

    run.log("Regularization:", np.float(args.C))
    run.log("Max Iterations:", np.str(args.max_iter))
    run.log("Penalty:", np.str(args.penalty))
    run.log("Solver:", np.str(args.solver))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter, penalty=args.penalty, solver=args.solver)

    scores = cross_val_score(model, x, y, scoring="accuracy", cv=cv, n_jobs=-1)
    run.log("Scores", scores)
    run.log("Accuracy", np.mean(scores))
    
    # train model on all samples and save it
    model.fit(x, y)
    joblib.dump(value=model, filename="./outputs/hd_model.pkl")

if __name__ == '__main__':
    main()
