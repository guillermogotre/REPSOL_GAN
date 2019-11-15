from invoke import task
import os
import json

@task
def run(c, config_path='./config.json', git=False, update=True):
    assert (os.path.isfile(config_path))
    with open(config_path,'r') as ifile:
        config_json = json.load(ifile)
    for p in [config_json['INDATA_FOLDER'],config_json['OUTDATA_FOLDER']]:
        if not os.path.isdir(p):
            os.makedirs(p)

    if git:
        c.run('git pull')

    if update:
        c.run('pip install -r requirements.txt --user')

    c.run("CUDA_VISIBLE_DEVICES={} python main.py".format(config_json['GPU_ID']))

