import os

REPO_PATH='path/2/language_modeling_via_stochastic_processes' # CHANGE ME! 
PATH2HUGGINGFACE='path/2/huggingface' # CHANGE ME! 

##################################################
###### The rest doesn't need to be changed! ###### 
##################################################

DATA_PATH=os.path.join(REPO_PATH, 'data')
PATH2TRANSFORMERS=os.path.join(REPO_PATH, 'language_modeling_via_stochastic_processes/transformers')
PATH2RECIPENLG=os.path.join(DATA_PATH, 'recipe_nlg')
PATH2WIKISECTION=os.path.join(DATA_PATH, 'wikisection')
PATH2WIKIHOW=os.path.join(DATA_PATH, 'wikihow', 'wiki_how_data.pkl')
PATH2TICKETTALK=os.path.join(DATA_PATH,'tickettalk')
PATH2TM2=os.path.join(DATA_PATH,'tm2')
PATH2ROCSTORIES=os.path.join(DATA_PATH,'roc_stories')

VISUALIZATION_DIR = os.path.join(
    REPO_PATH,
    'language_modeling_via_stochastic_processes/visualizations')

NAME2PRETRAINEDMODELPATH = {
    # wikisection
    "wikisection_cl_8": os.path.join(
        REPO_PATH,
        "models/wikisection/tc8/epoch=99-step=127199.ckpt"),
    "wikisection_cl_16": os.path.join(
        REPO_PATH,
        "models/wikisection/tc16/epoch=99-step=127199.ckpt"),
    "wikisection_cl_32": os.path.join(
        REPO_PATH,
        "models/wikisection/tc32/epoch=99-step=127199.ckpt"),
    # wikihow
    "wikihow_cl_8": os.path.join(
        REPO_PATH,
        "models/wikihow/tc8/epoch=99-step=75299.ckpt"),
    "wikihow_cl_16": os.path.join(
        REPO_PATH,
        "models/wikihow/tc16/epoch=99-step=75299.ckpt"),
    "wikihow_cl_32": os.path.join(
        REPO_PATH,
        "models/wikihow/tc32/epoch=99-step=75299.ckpt"),
    # tm2
    "tm2_cl_8": os.path.join(
        REPO_PATH,
        "models/tm2/tc8/epoch=99-step=78099.ckpt"),
    "tm2_cl_16": os.path.join(
        REPO_PATH,
        "models/tm2/tc16/epoch=99-step=78099.ckpt"),
    "tm2_cl_32": os.path.join(
        REPO_PATH,
        "models/tm2/tc32/epoch=99-step=78099.ckpt"),
    # tickettalk
    "tickettalk_cl_8": os.path.join(
        REPO_PATH,
        "models/tickettalk/tc8/epoch=99-step=78099.ckpt"),
    "tickettalk_cl_16": os.path.join(
        REPO_PATH,
        "models/tickettalk/tc16/epoch=99-step=78099.ckpt"),
    "tickettalk_cl_32": os.path.join(
        REPO_PATH,
        "models/tickettalk/tc32/epoch=99-step=78099.ckpt"),
    # recipe
    "recipe_cl_8": os.path.join(
        REPO_PATH,
        "models/recipe/tc8/epoch=99-step=21999.ckpt"),
    "recipe_cl_16": os.path.join(
        REPO_PATH,
        "models/recipe/tc16/epoch=99-step=21999.ckpt"),
    "recipe_cl_32": os.path.join(
        REPO_PATH,
        "models/recipe/tc32/epoch=99-step=21999.ckpt"),
    # rocstories
    "rocstories_cl_8": os.path.join(
        REPO_PATH,
        "models/rocstories/tc8/epoch=29-step=459869.ckpt"),
    "rocstories_cl_16": os.path.join(
        REPO_PATH,
        "models/rocstories/tc16/epoch=29-step=459869.ckpt"),
    "rocstories_cl_32": os.path.join(
        REPO_PATH,
        "models/rocstories/tc32/epoch=29-step=459869.ckpt"),
}
