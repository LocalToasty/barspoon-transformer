import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import exists
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Generate stats.')
    
    parser.add_argument('-r','--run-path',type=str,required=True)
    
    args = parser.parse_args()
    
    #run_path = r"/scratch/ws/1/s1787956-tim-cpath/TCGA-results/multilabel-zoom/d_model=512-dim_feedforward=2048-instances_per_bag=2048-learning_rate=0.0001-num_decoder_heads=8-num_encoder_heads=8-num_heads=8-num_layers=2"
    
    args.run_path = f"{args.run_path}/{os.listdir(args.run_path)[0]}"
    print(f"collecting stats for results of: {args.run_path.split('/')[-1]}")
    result_df = pd.DataFrame()
    
    for root, directories, files in os.walk(args.run_path):
        
        for directory in directories:
            if "figs" not in directory:
                tmp_df = pd.read_csv(os.path.join(root,f"{str(directory)}/lightning_logs/version_0/metrics.csv"))
                if len(result_df) == 0:
                    result_df = pd.DataFrame(columns=["fold"]+[c for c in tmp_df.columns if "test_" in c and "_auroc" in c])
                entry_list = [str(directory).split("=")[-1]]
                for c in list(result_df.columns[1:]):
                    if c in list(tmp_df.columns):
                        entry_list.append(tmp_df[c].values[-1])
                if len(entry_list)==len(list(result_df.columns)):
                    new_line = pd.Series(entry_list,index=result_df.columns)
                #result_df = result_df.append(new_line, ignore_index=True)
                result_df.loc[len(result_df)] = new_line

                #result_df = pd.concat([result_df,new_line],ignore_index=True)
        break
    print(result_df)
    
    result_df = result_df.sort_values(by="fold")
    
    avg_line = ["AVG"] + [np.mean(result_df[c].values) for c in list(result_df.columns)[1:]]
    avg_line = pd.Series(avg_line,index=result_df.columns)
    result_df.loc[len(result_df)] = avg_line
    result_df.to_csv(f"{args.run_path}/test_results.csv",index=False)
    if not exists(f"{args.run_path}/figs"):
        os.mkdir(f"{args.run_path}/figs")
    
    plt.figure(figsize=(26,14))
    
    for i,t in enumerate(result_df.columns[1:]):
        
        plt.plot(result_df["fold"],result_df[t],ls="dotted",marker='x',ms=8,label=f"{t.split('_')[1]}:{np.mean(result_df[t]):.3f}$\pm${np.std(result_df[t]):.3f}")
        
    plt.title("Targets",fontsize=28)
    plt.legend(fontsize=16)
    plt.xlabel("Fold",fontsize=24)
    plt.ylabel("AUROC",fontsize=24)
    plt.tick_params(axis='both', which='both', labelsize=22)
    plt.savefig(f"{args.run_path}/figs/aurocs.pdf",dpi=300)
    

    
