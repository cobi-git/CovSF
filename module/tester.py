# import sys
# sys.path.append('/home/qotmd01/CovSF_2/')
# sys.path.append('/home/qotmd01/CovSF_2/module/')

# from pathlib import Path
# import concurrent.futures

# import torch
# import torch.optim as optim


# if torch.cuda.is_available:
#     DEVICE = torch.device("cuda")
# else:
#     DEVICE = torch.device("cpu")

# MODEL = {
#     'Seq2Seq' : Seq2Seq
# }

# MAX_WORKER = 5

# TRAINED_PATH = Path('/home/qotmd01/CovSF_2/trained')

# class Validator():
#     def __init__(self,dataset_dir,save_name,model="Seq2Seq",n_folds=10):
#         self.dataset_dir = Path(dataset_dir)
#         self.save_name = save_name

#         self.save_path = TRAINED_PATH/save_name

    
#     def valid(self):
#         with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
#             futures = []

#         return
    
#     def _valid(self):
