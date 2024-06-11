import numpy as np
import pandas as pd
from itertools import combinations

def ensemble(data_path_list: dict, idx: tuple) -> None:
    output_list = []
    name = ""
    for idx in idxs:
        df = pd.read_csv(list(data_path_list.keys())[idx])
        output_list.append(np.array(df))
        name = name+'_'+list(data_path_list.values())[idx]

    esnb_result = []
    for i in range(len(output_list[0])):
        average = sum([output[i][1] / len(output_list) for output in output_list])
        esnb_result.append(round(average,1))

    esnb_dataframe = pd.DataFrame({"id": output_list[0][:, 0], "target": esnb_result})
    esnb_dataframe.to_csv(f"/data/ephemeral/home/level1/ksy/outputs/output{name}.csv", index=False)

if __name__=="__main__":

    data_path_list = {"/data/ephemeral/home/level1/ksy/outputs/output_koelectra.csv":"koelectra",
                      '/data/ephemeral/home/level1/ksy/outputs/output_roberta.csv':"roberta",
                      '/data/ephemeral/home/level1/ksy/outputs/output_rurupang.csv':"rurupang",
                    #   '/data/ephemeral/home/level1/ksy/outputs/output_roberta.csv':"roberta",
                    #   '/data/ephemeral/home/level1/ksy/outputs/output_roberta.csv':"roberta"
                    }   
    
    for idxs in combinations(range(3),3):
        ensemble(data_path_list,idxs)