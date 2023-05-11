#!/bin/bash

project="ICLR2022_v3_decoder"
tag="randDecoder"
env_name="language_modeling_via_stochastic_processes"
path2repo="/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder"
nseed=3

for seed in $(seq 1 1 $nseed); do
    ####### GPT2 ########
    # wikisection
    latent_dim=32 # This is just a dummy flag
    nlprun -n gpt2 -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name wikisection_filter --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_toyWiki_gpt2_seed'${seed}' --latent_dim='${latent_dim}' --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

    # wikihow
    nlprun -n gpt2 -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name wikihow --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_wikihow_gpt2'${latent_dim}'_seed'${seed}'  --latent_dim='${latent_dim}'  --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

    # tickettalk
    nlprun -n gpt2 -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name taskmaster --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_taskmaster_movies_gpt2'${latent_dim}'_seed'${seed}' --latent_dim='${latent_dim}'  --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

    # recipe
    nlprun -n gpt2 -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name recipe --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_recipe_gpt2'${latent_dim}'_seed'${seed}' --latent_dim='${latent_dim}'  --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

    # tm2
    nlprun -n gpt2 -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name restaurant_taskmaster --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_taskmaster_restaurant_gpt2'${latent_dim}'_seed'${seed}'  --latent_dim='${latent_dim}'  --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}


    ###### GPT2 sec dense and sparse ######
    nlprun -n gpt2 -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name wikisection_filter --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_toyWiki_gpt2_secDense_seed'${seed}' --latent_dim='${latent_dim}' --evaluation_strategy=steps --eval_steps=1000 --use_section_ids' -a ${env_name}

    nlprun -n gpt2 -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name wikisection_filter --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_toyWiki_gpt2_secSparse_seed'${seed}' --latent_dim='${latent_dim}' --evaluation_strategy=steps --eval_steps=1000 --use_section_ids --use_section_null' -a ${env_name}


    #### VAE ########
    for latent_dim in {8,16,32}; do
        # wikisection
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name wikisection_filter --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_toyWiki_vae'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/wikisection'${latent_dim}'_vae/checkpoints/epoch=99-step=127199.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # # wikihow
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name wikihow --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_wikihow_vae'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/wikihow'${latent_dim}'_vae/checkpoints/epoch=99-step=150899.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # tickettalk
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name taskmaster --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_taskmaster_movies_vae'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/taskmaster_movies'${latent_dim}'_vae/checkpoints/epoch=99-step=78099.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # recipe
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name recipe --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_recipe_vae'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/recipe'${latent_dim}'_vae/checkpoints/epoch=99-step=21999.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # tm2
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name restaurant_taskmaster --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_taskmaster_restaurant_vae'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/tmRestaurant'${latent_dim}'_vae/checkpoints/epoch=99-step=78099.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}
    done


    ###### ID ########
    for latent_dim in {8,16,32}; do
        # wikisection
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name wikisection_filter --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_toyWiki_infonce'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/wikisection'${latent_dim}'_infonce/checkpoints/epoch=99-step=127199.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # wikihow
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name wikihow --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_wikihow_infonce'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/wikihow'${latent_dim}'_infonce/checkpoints/epoch=99-step=150899.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # tickettalk
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name taskmaster --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_taskmaster_movies_infonce'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/taskmaster_movies'${latent_dim}'_infonce/checkpoints/epoch=99-step=78099.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # recipe
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name recipe --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_recipe_infonce'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/recipe'${latent_dim}'_infonce/checkpoints/epoch=99-step=21999.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # tm2
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name restaurant_taskmaster --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_taskmaster_restaurant_infonce'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/tmRestaurant'${latent_dim}'_infonce/checkpoints/epoch=99-step=78099.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}
    done



    ###### BROWNIAN ########
    for latent_dim in {8,16,32}; do
        # # wikisection
        nlprun -n brownian -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name wikisection_filter --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_toyWiki_brownian'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/wikisection'${latent_dim}'_brownian/checkpoints/epoch=99-step=127199.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # # wikihow
        nlprun -n brownian -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name wikihow --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_wikihow_brownian'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/wikihow'${latent_dim}'_brownian/checkpoints/epoch=99-step=150899.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # # tickettalk
        nlprun -n brownian -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name taskmaster --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_taskmaster_movies_brownian'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/taskmaster_movies'${latent_dim}'_brownian/checkpoints/epoch=99-step=78099.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # recipe
        nlprun -n brownian -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name recipe --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_recipe_brownian'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/recipe'${latent_dim}'_brownian/checkpoints/epoch=99-step=21999.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # # tm2
        nlprun -n brownian -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name restaurant_taskmaster --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_taskmaster_restaurant_brownian'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/tmRestaurant'${latent_dim}'_brownian/checkpoints/epoch=99-step=78099.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}
    done



    ###### TC ########
    for latent_dim in {8,16,32}; do
        # # wikisection
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name wikisection_filter --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_toyWiki_tc'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/wikisection'${latent_dim}'_tc/checkpoints/epoch=99-step=127199.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # # wikihow
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name wikihow --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_wikihow_tc'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/wikihow'${latent_dim}'_tc/checkpoints/epoch=99-step=150899.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # # tickettalk
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name taskmaster --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_taskmaster_movies_tc'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/taskmaster_movies'${latent_dim}'_tc/checkpoints/epoch=99-step=78099.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # recipe
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name recipe --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_recipe_tc'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/recipe'${latent_dim}'_tc/checkpoints/epoch=99-step=21999.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}

        # # tm2
        nlprun -n vae -g 1 'python run_time_clm.py --model_name_or_path gpt2 --dataset_name restaurant_taskmaster --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --project='${project}' --tag='${tag}' --seed='${seed}' --output_dir LM_taskmaster_restaurant_tc'${latent_dim}'_seed'${seed}' --cl_fpath=/juice/scr/rewang/nonstationarity_final/code/encoder_models/tmRestaurant'${latent_dim}'_tc/checkpoints/epoch=99-step=78099.ckpt --latent_dim='${latent_dim}' --use_contrastive_embeddings --evaluation_strategy=steps --eval_steps=1000' -a ${env_name}
    done
done
