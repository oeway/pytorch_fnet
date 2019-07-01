import os
import json


from fnet.cli.init import save_default_train_options

###################################################
# Download the 3D multi-channel tiffs via Quilt/T4
###################################################

gpu_id = 0

n_images_to_download = 40  # more images the better
train_fraction = 0.75

model_save_dir = "{}/model/".format(os.getcwd())
prefs_save_path = "{}/prefs.json".format(model_save_dir)



################################################
# Run the label-free stuff (dont change this)
################################################

save_default_train_options(prefs_save_path)

with open(prefs_save_path, "r") as fp:
    prefs = json.load(fp)

prefs["n_iter"] = 50000  # takes about 16 hours, go up to 250,000 for full training
prefs["interval_checkpoint"] = 10000

prefs["dataset_train"] = "fnet.data.HPAOnlineDataset"
prefs["dataset_train_kwargs"] = {}
prefs["dataset_val"] = "fnet.data.HPAOnlineDataset"
prefs["dataset_val_kwargs"] = {}
prefs["bpds_kwargs"] = {"patch_shape": [
    128,
    128
]}
prefs["fnet_model_kwargs"]  = {
    'criterion_class': 'torch.nn.MSELoss',
    'nn_class': 'fnet.nn_modules.fnet_nn_2d.Net',
}

prefs["nn_kwargs"]  = {
    'in_channels': 2,
    'out_channels': 1
}


# This Fnet call will be updated as a python API becomes available

with open(prefs_save_path, "w") as fp:
    json.dump(prefs, fp)

command_str = "fnet train {} --gpu_ids {}".format(prefs_save_path, gpu_id)

print(command_str)
os.system(command_str)
