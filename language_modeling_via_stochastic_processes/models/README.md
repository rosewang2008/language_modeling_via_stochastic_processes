Models should be put into this directory.

**NOTE: I'm still figuring out where to upload the pretrained encoders (~100GB) cost-free and directly from the compute cluster I'm using (rather than scp-ing). Until then, you'll need to train the encoders from scratch...if folks have suggestions, don't hesitate to reach out! I want to make the code as accessible as possible. :)**

The models are stored per dataset, like: `models/<dataset_name>/`. The current datasets supported are `wikisection`, `wikihow`, `recipe_nlg`, `roc_stories`, `tm2` and `tickettalk`.

The format of this directory should look something like this: 

```
>> ls path/2/repo/language_modeling_via_stochastic_processes/models/wikisection/*

wikisection/brownian16:
epoch=99-step=127199.ckpt

wikisection/brownian32:
epoch=99-step=127199.ckpt

wikisection/brownian8:
epoch=99-step=127199.ckpt

wikisection/infonce16:
epoch=99-step=127199.ckpt

wikisection/infonce32:
epoch=99-step=127199.ckpt

wikisection/infonce8:
epoch=99-step=127199.ckpt

wikisection/tc16:
epoch=99-step=127199.ckpt

wikisection/tc32:
epoch=99-step=127199.ckpt

wikisection/tc8:
epoch=99-step=127199.ckpt

wikisection/vae16:
epoch=99-step=127199.ckpt

wikisection/vae32:
epoch=99-step=114479.ckpt

wikisection/vae8:
epoch=99-step=127199.ckpt

```