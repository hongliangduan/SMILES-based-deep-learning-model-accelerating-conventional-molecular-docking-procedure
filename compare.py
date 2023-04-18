from tqdm import tqdm
import pandas as pd
import argparse
from pathlib import Path

def get_argments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_path", 
        default="output/Enamine_(pair,none,threshold,100,100,0.01,100)/9.0/1/636_iter_5.csv" ,
        help="the path of file that save the prediction"
    )
    parser.add_argument(
        "--score_path", 
        default="molpal-main/data/Enamine10k_scores.csv",
        help="the path of the scores of molecular pool"
    )
    parser.add_argument(
        "--output_file", 
        default="output/Enamine_(pair,none,threshold,100,100,0.01,100)/9.0/1/output.txt",
        help="path to save the output csv files"
    )
    parser.add_argument(
        "--topn", 
        default='100',
    )
    return parser.parse_args()

def main():
    args = get_argments()

    pred_path = args.pred_path
    score_path = args.score_path
    output_file = args.output_file
    topn = int(args.topn)

    pre = []
    data = []
    top = []

    if Path(pred_path).suffix == ".csv" and Path(score_path).suffix == ".csv":
        pred_data = pd.read_csv(pred_path)
        score_data = pd.read_csv(score_path)

        for i in tqdm(pred_data['smiles']):
            try:
                pre.append(i.strip())
            except:
                pre.append("nan")

        for i in tqdm(score_data['smiles']):
            data.append(i.strip())
    elif Path(pred_path).suffix != ".csv":
        raise TypeError("the pred_file isn't csv")
    elif Path(score_path).suffix != ".csv":
        raise TypeError("the score_file isn't csv")
    else:
        raise TypeError("the score_file and pred_file isn't csv")

    for i in range(topn):
        top.append(data[i])

    set1 = set(pre)
    set2 = set(top)
    set3 = set1 & set2

    with open(output_file, 'a') as output_file:
        output_file.write(f'{"%.2f"% ((len(set3)/topn)*100)}\n')

if __name__ == '__main__':
    main()