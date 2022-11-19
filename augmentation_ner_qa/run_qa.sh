# python -m torch.distributed.launch --nproc_per_node 2 --use_env run_ner.py
CUDA_VISIBLE_DEVICES=1 python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir tmp/debug_squad1 \
  --overwrite_output_dir \
  --num_train_epochs 50 \
  --n_train 50 \
  --aug_file squad_first50_aug3_genius \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --report_to none




# # for SQuAD-v2:
# CUDA_VISIBLE_DEVICES=1 python run_qa.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name squad_v2 \
#   --version_2_with_negative \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir saved_models/squad2_full_baseline \
#   --overwrite_output_dir \
#   --num_train_epochs 2 \


