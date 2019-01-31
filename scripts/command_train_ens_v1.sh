#/bin/bash
for version in B C
do
  python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir train/log_ens_v1_${version} --num_point 1024 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --ensemble_config 111
  python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir train/log_ens_v1_${version} --num_point 1024 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --restore_model_path train/log_ens_v1_${version}/model.ckpt --ensemble_config 222
  python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir train/log_ens_v1_${version} --num_point 1024 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --restore_model_path train/log_ens_v1_${version}/model.ckpt --ensemble_config 333
