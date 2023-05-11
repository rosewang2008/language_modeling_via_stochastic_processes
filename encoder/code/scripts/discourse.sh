#!/bin/bash

nseed=3

project="ICLR2022_v3_discourse"
env_name="language_modeling_via_stochastic_processes"
priority='low'
tag="tag"


# ============= GPT2 ============
for seed in $(seq 1 1 $nseed); do
    for k in {5,10}; do # dataset k
        #wikisection
        nlprun -n gpt2 -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=wikisectionK'${k}'_gpt2 --encoder=gpt2 --dataset=wikisection --batch-size=16 --use-section-ids --k='${k}' --project='${project}' --tag='${tag}' --seed='${seed} -a ${env_name}
        # tm2
        nlprun -n gpt2 -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=tmRestaurantK'${k}'_gpt2 --encoder=gpt2 --dataset=taskmaster --tm-type=restaurant --batch-size=16 --use-section-ids --project='${project}' --tag='${tag}'  --k='${k}' --seed='${seed} -a ${env_name}
        #tickettalgpt2
        nlprun -n gpt2 -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=ticketTalkK'${k}'_gpt2 --encoder=gpt2 --dataset=taskmaster --batch-size=8 --use-section-ids --k='${k}' --project='${project}' --tag='${tag}'  --seed='${seed} -a ${env_name}
    done
done
# ======================================


# ============= ALBERT ============
for seed in $(seq 1 1 $nseed); do
    for k in {5,10}; do # dataset k
        # #wikisection
        nlprun -n albert -g 1 'cd ..; python scripts/run_discourse.py --config="config/albert_discourse.json" --exp-name=wikisectionK'${k}'_albert --encoder=albert --dataset=wikisection --batch-size=16 --use-section-ids --k='${k}' --tag='${tag}' --project='${project}'  --seed='${seed} -a ${env_name}
        # tm2
        nlprun -n aalbert -g 1 'cd ..; python scripts/run_discourse.py --config="config/albert_discourse.json" --exp-name=tmRestaurantK'${k}'_albert --encoder=albert --dataset=taskmaster --tm-type=restaurant --batch-size=16 --use-section-ids --k='${k}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
        #tickettalk
        nlprun -n albert -g 1 'cd ..; python scripts/run_discourse.py --config="config/albert_discourse.json" --exp-name=ticketTalkK'${k}'_albert --encoder=albert --dataset=taskmaster --batch-size=8 --use-section-ids --k='${k}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
    done
done
# ======================================

#

# ============= BERT ============
 for seed in $(seq 1 1 $nseed); do
     for k in {5,10}; do # dataset k
         #wikisection
         nlprun -n bert -g 1 'cd ..; python scripts/run_discourse.py --config="config/bert_discourse.json" --exp-name=wikisectionK'${k}'_bert --encoder=bert --dataset=wikisection --batch-size=16 --use-section-ids --k='${k}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
         # tm2
         nlprun -n bert -g 1 'cd ..; python scripts/run_discourse.py --config="config/bert_discourse.json" --exp-name=tmRestaurantK'${k}'_bert --encoder=bert --dataset=taskmaster --tm-type=restaurant --batch-size=16 --use-section-ids --k='${k}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
         # ickettal
         nlprun -n bert -g 1 'cd ..; python scripts/run_discourse.py --config="config/bert_discourse.json" --exp-name=ticketTalkK'${k}'_bert --encoder=bert --dataset=taskmaster --batch-size=8 --use-section-ids --k='${k}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
     done
 done
# ======================================



# ============= S-BERT ============
for seed in $(seq 1 1 $nseed); do
    for k in {5,10}; do # dataset k
        #wikisection
        nlprun -n sbert -g 1 'cd ..; python scripts/run_discourse.py --config="config/bert_discourse.json" --exp-name=wikisectionK'${k}'_sbert --encoder=sbert --dataset=wikisection --batch-size=16 --use-section-ids --k='${k}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
        # tm2
        nlprun -n  d_sbert -g 1 'cd ..; python scripts/run_discourse.py --config="config/bert_discourse.json" --exp-name=tmRestaurantK'${k}'_sbert --encoder=sbert --dataset=taskmaster --tm-type=restaurant --batch-size=16 --use-section-ids --k='${k}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
        #tickettal
        nlprun -n sbert -g 1 'cd ..; python scripts/run_discourse.py --config="config/bert_discourse.json" --exp-name=ticketTalkK'${k}'_sbert --encoder=sbert --dataset=taskmaster --batch-size=8 --use-section-ids --k='${k}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
    done
done
======================================




# ============= Sim-CSE ============
for seed in $(seq 1 1 $nseed); do
    for k in {5,10}; do # dataset k
        #wikisection
        nlprun -n simcse -g 1 'cd ..; python scripts/run_discourse.py --config="config/bert_discourse.json" --exp-name=wikisectionK'${k}'_simcse --encoder=simcse --dataset=wikisection --batch-size=16 --use-section-ids --k='${k}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
        # # tm2
        nlprun -n simcse -g 1 'cd ..; python scripts/run_discourse.py --config="config/bert_discourse.json" --exp-name=tmRestaurantK'${k}'_simcse --encoder=simcse --dataset=taskmaster --tm-type=restaurant --batch-size=16 --use-section-ids --k='${k}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
        # #tickettal
        nlprun -n simcse -g 1 'cd ..; python scripts/run_discourse.py --config="config/bert_discourse.json" --exp-name=ticketTalkK'${k}'_simcse --encoder=simcse --dataset=taskmaster --batch-size=8 --use-section-ids --k='${k}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
    done
done
# ======================================

# ============= VAE ============
for seed in $(seq 1 1 $nseed); do
    for k in {5,10}; do # dataset k
        for latent_dim in {8,16,32}; do
            #wikisection
            nlprun -n  d_vae -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=wikisectionK'${k}'_vae --encoder=cl --dataset=wikisection --batch-size=16 --use-section-ids --k='${k}' --fpath=wikisection'${latent_dim}'_vae/checkpoints/epoch=99-step=127199.ckpt --latent-dim='${latent_dim}' --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
            # tm2
            nlprun -n  d_vae -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=tmRestaurantK'${k}'_vae --encoder=cl --dataset=taskmaster --tm-type=restaurant --batch-size=16 --use-section-ids --k='${k}' --fpath=tmRestaurant'${latent_dim}'_vae/checkpoints/epoch=99-step=78099.ckpt  --tag='${tag}' --project='${project}' --latent-dim='${latent_dim}' --seed='${seed} -a ${env_name}
            #tickettalvae
            nlprun -n  d_vae -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=ticketTalkK'${k}'_vae --encoder=cl --dataset=taskmaster --batch-size=8 --use-section-ids --fpath=taskmaster_movies'${latent_dim}'_vae/checkpoints/epoch=99-step=78099.ckpt --k='${k}'  --tag='${tag}' --project='${project}' --latent-dim='${latent_dim}' --seed='${seed} -a ${env_name}
        done
    done
done
# ======================================

#
# ============= ID ============
for seed in $(seq 1 1 $nseed); do
    for k in {5,10}; do # dataset k
        for latent_dim in {8,16,32}; do
            #wikisection
            nlprun -n  d_infonce -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=wikisectionK'${k}'_infonce --encoder=cl --dataset=wikisection --batch-size=16 --use-section-ids --k='${k}' --fpath=wikisection'${latent_dim}'_infonce/checkpoints/epoch=99-step=127199.ckpt --latent-dim='${latent_dim}' --tag='${tag}' --project='${project}'  --seed='${seed} -a ${env_name}
            # # tm2
            nlprun -n  d_infonce -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=tmRestaurantK'${k}'_infonce --encoder=cl --dataset=taskmaster --tm-type=restaurant --batch-size=16 --use-section-ids --k='${k}' --fpath=tmRestaurant'${latent_dim}'_infonce/checkpoints/epoch=99-step=78099.ckpt  --tag='${tag}' --project='${project}'  --latent-dim='${latent_dim}' --seed='${seed} -a ${env_name}
            # #tickettalinfonce
            nlprun -n  d_infonce -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=ticketTalkK'${k}'_infonce --encoder=cl --dataset=taskmaster --batch-size=8 --use-section-ids --fpath=taskmaster_movies'${latent_dim}'_infonce/checkpoints/epoch=99-step=78099.ckpt --k='${k}'  --latent-dim='${latent_dim}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
        done
    done
done
# ======================================



# ============= BM ============
for seed in $(seq 1 1 $nseed); do
    for k in {5,10}; do # dataset k
        for latent_dim in {8,16,32}; do
            #wikisection
            nlprun -n  d_brownian -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=wikisectionK'${k}'_brownian --encoder=cl --dataset=wikisection --batch-size=16 --use-section-ids --k='${k}' --fpath=wikisection'${latent_dim}'_brownian/checkpoints/epoch=99-step=127199.ckpt  --latent-dim='${latent_dim}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
            # # tm2
            nlprun -n  d_brownian -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=tmRestaurantK'${k}'_brownian --encoder=cl --dataset=taskmaster --tm-type=restaurant --batch-size=16 --use-section-ids --k='${k}' --fpath=tmRestaurant'${latent_dim}'_brownian/checkpoints/epoch=99-step=78099.ckpt  --tag='${tag}' --project='${project}'  --latent-dim='${latent_dim}' --seed='${seed} -a ${env_name}
            # #tickettalbrownian
            nlprun -n d_brownian -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=ticketTalkK'${k}'_brownian --encoder=cl --dataset=taskmaster --batch-size=8 --use-section-ids --fpath=taskmaster_movies'${latent_dim}'_brownian/checkpoints/epoch=99-step=78099.ckpt --k='${k}'  --latent-dim='${latent_dim}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
        done
    done
done
# ======================================


# ============= TC ============
for seed in $(seq 1 1 $nseed); do
    for k in {5,10}; do # dataset k
        for latent_dim in {8,16,32}; do
            #wikisection
            nlprun -n tc -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=wikisectionK'${k}'_tc --encoder=cl --dataset=wikisection --batch-size=16 --use-section-ids --k='${k}'  --latent-dim='${latent_dim}' --fpath=wikisection'${latent_dim}'_tc/checkpoints/epoch=99-step=127199.ckpt  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
            # tm2
            nlprun -n tc -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=tmRestaurantK'${k}'_tc --encoder=cl --dataset=taskmaster --tm-type=restaurant --batch-size=16 --use-section-ids --k='${k}'  --latent-dim='${latent_dim}' --fpath=tmRestaurant'${latent_dim}'_tc/checkpoints/epoch=99-step=78099.ckpt  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
            #tickettaltc
            nlprun -n tc -g 1 'cd ..; python scripts/run_discourse.py --config="config/gpt2_discourse.json" --exp-name=ticketTalkK'${k}'_tc --encoder=cl --dataset=taskmaster --batch-size=8 --use-section-ids  --latent-dim='${latent_dim}' --fpath=taskmaster_movies'${latent_dim}'_tc/checkpoints/epoch=99-step=78099.ckpt --k='${k}'  --tag='${tag}' --project='${project}' --seed='${seed} -a ${env_name}
        done
    done
done
# # ======================================

