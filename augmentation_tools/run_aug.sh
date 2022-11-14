#  5huffpost_100 5huffpost_200 5huffpost_500 5huffpost_1000
#  sst2_50 sst2_100 sst2_200 sst2_500 sst2_1000
# imdb_50 imdb_100 imdb_200 imdb_500 imdb_1000
# yahoo10k_50 yahoo10k_100 yahoo10k_200 yahoo10k_500 yahoo10k_1000
# 20ng_50 20ng_100 20ng_200 20ng_500 20ng_1000
# yahooA10k_100 yahooA10k_200 yahooA10k_500 yahooA10k_1000

# python backtrans_clf.py --dataset_name sst2_50 --inter_langs de-zh-fr-es --n_aug 4
# python conditional_mlm_clf.py --dataset_name sst2_100 --mlm_model_path ../saved_models/c-mlm/sst2_50_roberta-large --n_aug 4
# python conditional_clm_clf.py --dataset_name sst2_50 --clm_model_path ../saved_models/c-clm/sst2_50_gpt2 --n_aug 4

CUDA_VISIBLE_DEVICES=3
# 5huffpost_100 5huffpost_200 5huffpost_500 5huffpost_1000
for dataset_name in 5huffpost_1000 5huffpost_500 5huffpost_200 5huffpost_100 5huffpost_50; do
    # # EDA:
    # python eda_clf.py --dataset_name $dataset_name --method mix --simdict wordnet --p 0.1 --n_aug 1

    # # STA:
    # python sta_clf.py --dataset_name $dataset_name --method mix --simdict wordnet --p 0.1 --n_aug 1

    # # BackTrans

    # python backtrans_clf.py --dataset_name $dataset_name --inter_langs de-zh-fr-es --n_aug 4

    # # MLM
    # python mlm_clf.py --dataset_name $dataset_name --mlm_model_path roberta-large --p 0.1 --topk 5 --n_aug 4

    # # C-MLM
    # python conditional_mlm_finetune.py --dataset_name $dataset_name --mlm_model_path roberta-large --num_train_epochs 15

    # python conditional_mlm_clf.py --dataset_name $dataset_name --mlm_model_path ../saved_models/c-mlm/${dataset_name}_roberta-large --n_aug 4

    # LAMBADA
    # python conditional_clm_finetune.py --clm_model_path gpt2 --dataset_name $dataset_name --num_train_epochs 15

    # python conditional_clm_clf.py --dataset_name $dataset_name --clm_model_path ../saved_models/c-clm/${dataset_name}_gpt2 --n_aug 1

    # SEGA
    # python sega_clf.py \
    #     --dataset_name $dataset_name \
    #     --sega_model_path beyond/sega-large \
    #     --template 4 \
    #     --sega_version sega-old-aspect_only \
    #     --aspect_only \
    #     --n_aug 4 \

    # python sega_clf.py \
    #     --dataset_name $dataset_name \
    #     --sega_model_path beyond/sega-large \
    #     --template 4 \
    #     --sega_version sega-old-no_aspect \
    #     --no_aspect \
    #     --n_aug 4 \

    # python sega_clf.py \
    #     --dataset_name $dataset_name \
    #     --sega_model_path ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
    #     --template 4 \
    #     --sega_version sega-old \
    #     --n_aug 4 \
    #     --add_prompt

    # # use finetuned-SEGA
    # python sega_finetune.py \
    #     --dataset_name $dataset_name \
    #     --checkpoint ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
    #     --max_num_sent 10 \
    #     --num_train_epochs 10 \
    #     --batch_size 16

    # python sega_clf.py \
    #     --dataset_name $dataset_name \
    #     --sega_model_path ../saved_models/sega_finetuned_for_$dataset_name \
    #     --template 4 \
    #     --sega_version sega-fine \
    #     --add_prompt \
    #     --n_aug 4 \

    # directly fine-tune BART
    # python sega_finetune.py \
    #     --dataset_name $dataset_name \
    #     --checkpoint facebook/bart-large\
    #     --comment _orig_bart \
    #     --max_num_sent 10 \
    #     --num_train_epochs 10 \
    #     --batch_size 16

    python sega_clf.py \
        --dataset_name $dataset_name \
        --sega_model_path ../saved_models/sega_finetuned_for_${dataset_name}_orig_bart \
        --template 4 \
        --sega_version bart-fine \
        --add_prompt \
        --n_aug 4 

    # directly use BART, no finetuning
    python sega_clf.py \
        --dataset_name $dataset_name \
        --sega_model_path facebook/bart-large \
        --template 4 \
        --sega_version poor-bart \
        --add_prompt \
        --n_aug 4 
done



# CUDA_VISIBLE_DEVICES=3
# for dataset_name in imdb_50 imdb_100 imdb_200 imdb_500 imdb_1000; do
#     # SEGA
#     # python sega_clf.py \
#     #     --dataset_name $dataset_name \
#     #     --sega_model_path ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
#     #     --template 4 \
#     #     --sega_version sega-old \
#     #     --n_aug 4 \

#     python sega_clf.py \
#         --dataset_name $dataset_name \
#         --sega_model_path ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
#         --template 4 \
#         --sega_version sega-old \
#         --max_length 100 \
#         --n_aug 4 \
#         --add_prompt

#     # use finetuned-SEGA
#     python sega_finetune.py \
#         --dataset_name $dataset_name \
#         --checkpoint ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
#         --max_num_sent 5 \
#         --num_train_epochs 15 \
#         --batch_size 16

#     python sega_clf.py \
#         --dataset_name $dataset_name \
#         --sega_model_path ../saved_models/sega_finetuned_for_$dataset_name \
#         --template 4 \
#         --sega_version sega-fine \
#         --max_length 100 \
#         --add_prompt \
#         --n_aug 4 
# done

# CUDA_VISIBLE_DEVICES=7
# for dataset_name in yahooA10k_50 yahooA10k_100 yahooA10k_200 yahooA10k_500 yahooA10k_1000; do
#     # SEGA
#     # python sega_clf.py \
#     #     --dataset_name $dataset_name \
#     #     --sega_model_path ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
#     #     --template 4 \
#     #     --sega_version sega-old \
#     #     --n_aug 4 \

#     python sega_clf.py \
#         --dataset_name $dataset_name \
#         --sega_model_path ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
#         --template 4 \
#         --sega_version sega-old \
#         --max_length 200 \
#         --n_aug 4 \
#         --add_prompt

#     # use finetuned-SEGA
#     # python sega_finetune.py \
#     #     --dataset_name $dataset_name \
#     #     --checkpoint ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
#     #     --max_num_sent 5 \
#     #     --num_train_epochs 15 \
#     #     --batch_size 16

#     python sega_clf.py \
#         --dataset_name $dataset_name \
#         --sega_model_path ../saved_models/sega_finetuned_for_$dataset_name \
#         --template 4 \
#         --sega_version sega-fine \
#         --max_length 200 \
#         --add_prompt \
#         --n_aug 4 
# done


# CUDA_VISIBLE_DEVICES=1
# for dataset_name in 20ng_50 20ng_100; do
#     # SEGA
#     # python sega_clf.py \
#     #     --dataset_name $dataset_name \
#     #     --sega_model_path ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
#     #     --template 4 \
#     #     --sega_version sega-old \
#     #     --n_aug 4 \

#     # python sega_clf.py \
#     #     --dataset_name $dataset_name \
#     #     --sega_model_path ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
#     #     --template 4 \
#     #     --sega_version sega-old \
#     #     --max_length 200 \
#     #     --n_aug 4 \
#     #     --add_prompt

#     # use finetuned-SEGA
#     python sega_finetune.py \
#         --dataset_name $dataset_name \
#         --checkpoint ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
#         --max_num_sent 5 \
#         --num_train_epochs 15 \
#         --batch_size 16

#     python sega_clf.py \
#         --dataset_name $dataset_name \
#         --sega_model_path ../saved_models/sega_finetuned_for_$dataset_name \
#         --template 4 \
#         --sega_version sega-fine \
#         --max_length 200 \
#         --add_prompt \
#         --n_aug 4 
# done


# CUDA_VISIBLE_DEVICES=0
# for dataset_name in sst2-l_200 sst2-l_500 sst2-l_1000; do
#     # SEGA
#     # python sega_clf.py \
#     #     --dataset_name $dataset_name \
#     #     --sega_model_path ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
#     #     --template 4 \
#     #     --sega_version sega-old \
#     #     --n_aug 4 \

#     python sega_clf.py \
#         --dataset_name $dataset_name \
#         --sega_model_path ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
#         --template 4 \
#         --sega_version sega-old \
#         --max_length 50 \
#         --n_aug 4 \
#         --add_prompt

    # use finetuned-SEGA
    # python sega_finetune.py \
    #     --dataset_name $dataset_name \
    #     --checkpoint ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
    #     --max_num_sent 5 \
    #     --num_train_epochs 15 \
    #     --batch_size 16

    # python sega_clf.py \
    #     --dataset_name $dataset_name \
    #     --sega_model_path ../saved_models/sega_finetuned_for_$dataset_name \
    #     --template 4 \
    #     --sega_version sega-fine \
    #     --max_length 60 \
    #     --add_prompt \
    #     --n_aug 4 
# done


# CUDA_VISIBLE_DEVICES=0
# # bbc_200 bbc_500 bbc_1000
# for dataset_name in bbc_50 bbc_100; do
#     # SEGA
#     # python sega_clf.py \
#     #     --dataset_name $dataset_name \
#     #     --sega_model_path ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
#     #     --template 4 \
#     #     --sega_version sega-old \
#     #     --n_aug 4 \

#     # python sega_clf.py \
#     #     --dataset_name $dataset_name \
#     #     --sega_model_path ../saved_models/bart-large-c4-l_50_200-d_13799838-yake_mask-t_3900800/checkpoint-152375 \
#     #     --template 4 \
#     #     --sega_version sega-old \
#     #     --max_length 200 \
#     #     --n_aug 4 \
#     #     --add_prompt

#     # use finetuned-SEGA
#     python sega_finetune.py \
#         --dataset_name $dataset_name \
#         --checkpoint ../saved_models/bart-base-c4-realnewslike-4templates-passage-and-sent-max15sents_2-sketch4/checkpoint-215625 \
#         --max_num_sent 5 \
#         --num_train_epochs 15 \
#         --batch_size 16

#     python sega_clf.py \
#         --dataset_name $dataset_name \
#         --sega_model_path ../saved_models/sega_finetuned_for_$dataset_name \
#         --template 4 \
#         --sega_version sega-fine \
#         --max_length 200 \
#         --add_prompt \
#         --n_aug 4 
# done




#### sega-mixup
# yahooA10k_50 20ng_50 bbc_50 5huffpost_50 imdb_50 sst2-l_50
# CUDA_VISIBLE_DEVICES=6
# for dataset_name in bbc_50; do
#     python sega_mixup_clf.py \
#         --dataset_name $dataset_name \
#         --max_ngram 3 \
#         --sketch_n_kws 15 \
#         --extract_global_kws \
#         --sega_version sega-mixup \
#         --n_aug 4
# done



### EDA/STA single operation:
# replace/delete/insert/swap/mix
# replace/delete/insert/positive/mix

# for operation in replace delete insert swap; do
#     python eda_clf.py --dataset_name 5huffpost_50 --method $operation \
#         --simdict wordnet --p 0.05 --n_aug 1
# done


# for operation in replace delete insert positive; do
#     python sta_clf.py --dataset_name 5huffpost_50 --method $operation \
#         --simdict wordnet --p 0.05 --n_aug 1
# done

# eda_insert_wordnet_0.1_1
# sta_positive_wordnet_0.1_1