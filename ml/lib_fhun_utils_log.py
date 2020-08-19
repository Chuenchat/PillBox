import os
import pandas as pd
from lib_utils_config_parse import cfg

def save_loss(path, epoch, loc_loss, conf_loss, lr=0):
    
    columns_list = ["epoch", "loc_loss", "conf_loss", "lr"]
    df_loss = pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame(data=[], columns = columns_list)
    d = {"epoch": [epoch],
         "loc_loss": [loc_loss],
         "conf_loss": [conf_loss],
         "lr": [lr]
         }
    df_loss = df_loss.append(pd.DataFrame(data=d, columns = columns_list), ignore_index=True)
    df_loss.to_csv(path, encoding='utf-8', index=False)


def save_mAP(path, epoch, mAP)  :
    
    columns_list = ["epoch", "mAP"]
    df_mAP = pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame(data=[], columns = columns_list)
    d = {"epoch": [epoch],
         "mAP": [mAP]
         }
    df_mAP = df_mAP.append(pd.DataFrame(data=d, columns = columns_list), ignore_index=True)    
    df_mAP.to_csv(path, encoding='utf-8', index=False)


def save_multiple_loss(path, epoch, loc_loss, conf_loss, occ_loss, lr=0):
    
    columns_list = ["epoch", "loc_loss", "conf_loss", "occ_loss", "lr"]
    df_loss = pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame(data=[], columns = columns_list)
    d = {"epoch": [epoch],
         "loc_loss": [loc_loss],
         "conf_loss": [conf_loss],
         "occ_loss": [occ_loss],
         "lr": [lr]
         }
    df_loss = df_loss.append(pd.DataFrame(data=d, columns = columns_list), ignore_index=True)
    df_loss.to_csv(path, encoding='utf-8', index=False)
