#!/bin/bash

nseed=1

project="ICLR2022_v2_encoder"
tag="tag"
environment_name="language_modeling_via_stochastic_processes"


# TIME CONTROL
for latent_dim in {8,16,32}; do
    # Wikisection
    nlprun -n wikisection -g 1 ' cd ..; python scripts/run_ou.py --config=config/bridge_wikisection.json --project='${project}' --use-section-ids --exp-name=wikisection'${latent_dim}'_tc --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=WikiRandomTData --num-epochs=100 --tag='${tag} -a ${environment_name}

    # Wikihow
    nlprun -n wikihow_encoder -g 1 ' cd ..; python scripts/run_ou.py --config=config/bridge_wikihow.json --project='${project}' --use-section-ids --exp-name=wikihow'${latent_dim}'_tc --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=WikihowRandomT --num-epochs=100' -a ${environment_name}

    # Tickettalk
    nlprun -n tm_movies -g 1 'cd ..; python scripts/run_ou.py --config=config/bridge_taskmaster.json --project='${project}' --use-section-ids --exp-name=taskmaster_movies'${latent_dim}'_tc --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=TaskmasterRandomT --num-epochs=100 --tag='${tag}  -a ${environment_name}

    # Restaurant
    nlprun -n tm_restaurant -g 1 ' cd ..; python scripts/run_ou.py --config="config/bridge_taskmaster.json" --project='${project}' --use-section-ids --exp-name=tmRestaurant'${latent_dim}'_tc --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=TaskmasterRandomT --tm-type=restaurant --num-epochs=100' -a ${environment_name}

    # Recipe
    nlprun -n recipe -g 1 ' cd ..; python scripts/run_ou.py --config="config/bridge_recipe_nlg.json" --project='${project}' --use-section-ids --exp-name=recipe'${latent_dim}'_tc --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=RecipeRandomT --num-epochs=100' -a ${environment_name}
done


# VAE
for latent_dim in {8,16,32}; do
     # Wikisection
     nlprun -n wikisection -g 1 ' cd ..; python scripts/run_ou.py --config=config/vae.json --project='${project}' --use-section-ids --exp-name=wikisection'${latent_dim}'_vae --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=WikiRandomTData --num-epochs=100 --tag='${tag}  -a ${environment_name}

     # Wikihow
     nlprun -n wikihow_encoder -g 1 ' cd ..; python scripts/run_ou.py --config=config/vae.json --project='${project}' --use-section-ids --exp-name=wikihow'${latent_dim}'_vae --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=WikihowRandomT --num-epochs=100' -a ${environment_name}

     # Tickettalk
     nlprun -n tm_movies -g 1 'cd ..; python scripts/run_ou.py --config=config/vae.json --project='${project}' --use-section-ids --exp-name=taskmaster_movies'${latent_dim}'_vae --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=TaskmasterRandomT --num-epochs=100 --tag='${tag}  -a ${environment_name}

     # Restaurant
     nlprun -n tm_restaurant -g 1 ' cd ..; python scripts/run_ou.py --config="config/vae.json" --project='${project}' --use-section-ids --exp-name=tmRestaurant'${latent_dim}'_vae --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=TaskmasterRandomT --tm-type=restaurant --num-epochs=100' -a ${environment_name}

     # Recipe
     nlprun -n recipe -g 1 ' cd ..; python scripts/run_ou.py --config="config/vae.json" --project='${project}' --use-section-ids --exp-name=recipe'${latent_dim}'_vae --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=RecipeRandomT --num-epochs=100' -a ${environment_name}
 done

 # Brownian
 for latent_dim in {8,16,32}; do
     # Wikisection
     nlprun -n wikisection -g 1 ' cd ..; python scripts/run_ou.py --config=config/bridge_wikisection.json --project='${project}' --use-section-ids --exp-name=wikisection'${latent_dim}'_brownian --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianRandomTLoss --dataset=WikiRandomTData --num-epochs=100' -a ${environment_name}

     # Wikihow
     nlprun -n wikihow_encoder -g 1 ' cd ..; python scripts/run_ou.py --config=config/bridge_wikihow.json --project='${project}' --use-section-ids --exp-name=wikihow'${latent_dim}'_brownian --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianRandomTLoss --dataset=WikihowRandomT --num-epochs=100' -a ${environment_name}

     # Tickettalk
     nlprun -n tm_movies -g 1 'cd ..; python scripts/run_ou.py --config=config/bridge_taskmaster.json --project='${project}' --use-section-ids --exp-name=taskmaster_movies'${latent_dim}'_brownian --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianRandomTLoss --dataset=TaskmasterRandomT --num-epochs=100' -a ${environment_name}

     # Restaurant
     nlprun -n tm_restaurant -g 1 ' cd ..; python scripts/run_ou.py --config="config/bridge_taskmaster.json" --project='${project}' --use-section-ids --exp-name=tmRestaurant'${latent_dim}'_brownian --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianRandomTLoss --dataset=TaskmasterRandomT --tm-type=restaurant --num-epochs=100' -a ${environment_name}

     # Recipe
     nlprun -n recipe -g 1 ' cd ..; python scripts/run_ou.py --config="config/bridge_recipe_nlg.json" --project='${project}' --use-section-ids --exp-name=recipe'${latent_dim}'_brownian --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianRandomTLoss --dataset=RecipeRandomT --num-epochs=100' -a ${environment_name}

 done

 # ID
 for latent_dim in {8,16,32}; do
     # Wikisection
     nlprun -n wikisection -g 1 ' cd ..; python scripts/run_ou.py --config=config/infonce.json --project='${project}' --use-section-ids --exp-name=wikisection'${latent_dim}'_infonce --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=WikiTPKData --num-epochs=100' -a ${environment_name}

     # Wikihow
     nlprun -n wikihow_encoder -g 1 ' cd ..; python scripts/run_ou.py --config=config/infonce.json --project='${project}' --use-section-ids --exp-name=wikihow'${latent_dim}'_infonce --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=WikihowTPKData --num-epochs=100' -a ${environment_name}

     # Tickettalk
     nlprun -n tm_movies -g 1 'cd ..; python scripts/run_ou.py --config=config/infonce.json --project='${project}' --use-section-ids --exp-name=taskmaster_movies'${latent_dim}'_infonce --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=TaskmasterTPKData --num-epochs=100' -a ${environment_name}

     # Restaurant
     nlprun -n tm_restaurant -g 1 ' cd ..; python scripts/run_ou.py --config="config/infonce.json" --project='${project}' --use-section-ids --exp-name=tmRestaurant'${latent_dim}'_infonce --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=TaskmasterTPKData --tm-type=restaurant --num-epochs=100' -a ${environment_name}

     # Recipe
     nlprun -n recipe -g 1 ' cd ..; python scripts/run_ou.py --config="config/infonce.json" --project='${project}' --use-section-ids --exp-name=recipe'${latent_dim}'_infonce --contrast=t --latent-dim='${latent_dim}' --k=1 --batch-size=32 --loss=BrownianBridgeRandomTLoss --dataset=RecipeTPKData --num-epochs=100' -a ${environment_name}
 done


