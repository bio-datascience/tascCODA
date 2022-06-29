from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent

def smillie() -> pd.DataFrame:
  """
  scRNA-seq data of the small intestine of mice under Ulcerative Colitis
  
  Smillie et al., 2019
 
  Returns
  -------
  data matrix as pandas data frame.
    
  """
  filename = HERE / 'smillie_UC_processed.csv'
  
  return pd.read_csv(filename)
