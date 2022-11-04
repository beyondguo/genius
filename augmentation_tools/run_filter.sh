# for dataset_name in sst2-l_50 sst2-l_100 sst2-l_200 sst2-l_500 sst2-l_1000; do
#     # filter SEGA
#     python aug_filter_clf.py --dataset_name $dataset_name --backbone distilbert-base-cased \
#         --aug_file_name sega_promptTrue_asonly_False_sega-old_aug4 --threshold 0.8

#     # filter SEGA-fine
#     python aug_filter_clf.py --dataset_name $dataset_name --backbone distilbert-base-cased \
#         --aug_file_name sega_promptTrue_asonly_False_sega-fine_aug4 --threshold 0.8
    
#     # filter LAMBADA
#     python aug_filter_clf.py --dataset_name $dataset_name --backbone distilbert-base-cased \
#         --aug_file_name cclm_${dataset_name}_gpt2_aug4 --threshold 0.8
# done

for dataset_name in imdb_200 imdb_500 imdb_1000; do
    # filter SEGA
    python aug_filter_clf.py --dataset_name $dataset_name --backbone distilbert-base-cased \
        --aug_file_name sega_promptTrue_asonly_False_sega-old_aug4 --threshold 0.7

    # filter SEGA-fine
    python aug_filter_clf.py --dataset_name $dataset_name --backbone distilbert-base-cased \
        --aug_file_name sega_promptTrue_asonly_False_sega-fine_aug4 --threshold 0.7
    
    # filter LAMBADA
    python aug_filter_clf.py --dataset_name $dataset_name --backbone distilbert-base-cased \
        --aug_file_name cclm_${dataset_name}_gpt2_aug4 --threshold 0.7
done

for dataset_name in yahooA10k_50 yahooA10k_100 yahooA10k_200 yahooA10k_500 yahooA10k_1000; do
    # filter SEGA
    python aug_filter_clf.py --dataset_name $dataset_name --backbone distilbert-base-cased \
        --aug_file_name sega_promptTrue_asonly_False_sega-old_aug4 --threshold 0.7

    # filter SEGA-fine
    python aug_filter_clf.py --dataset_name $dataset_name --backbone distilbert-base-cased \
        --aug_file_name sega_promptTrue_asonly_False_sega-fine_aug4 --threshold 0.7
    
    # filter LAMBADA
    python aug_filter_clf.py --dataset_name $dataset_name --backbone distilbert-base-cased \
        --aug_file_name cclm_${dataset_name}_gpt2_aug4 --threshold 0.7
done


for dataset_name in bbc_500 bbc_1000; do
    # filter SEGA
    python aug_filter_clf.py --dataset_name $dataset_name --backbone distilbert-base-cased \
        --aug_file_name sega_promptTrue_asonly_False_sega-old_aug4 --threshold 0.7

    # filter SEGA-fine
    python aug_filter_clf.py --dataset_name $dataset_name --backbone distilbert-base-cased \
        --aug_file_name sega_promptTrue_asonly_False_sega-fine_aug4 --threshold 0.7
    
    # filter LAMBADA
    python aug_filter_clf.py --dataset_name $dataset_name --backbone distilbert-base-cased \
        --aug_file_name cclm_${dataset_name}_gpt2_aug4 --threshold 0.7
done