# no aug
# CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --num_iter 5 --bsz 16

# # K2T
# CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file mix1_doc_aspect_avg_with_label  --num_iter 5 --bsz 16
# CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file mix4_doc_aspect_avg_with_label  --num_iter 5 --bsz 16

# # random, 4 times
# CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file random_re_in_sw_de_0.1_1/train_mix  --num_iter 5 --bsz 16

# # # selective global, 4 times
# CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file selective_global_re_in_se_de_0.05_Q2_1/train_mix --num_iter 5 --bsz 16
# CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file selective_global_re_in_se_de_0.1_Q2_1/train_mix --num_iter 5 --bsz 16
# CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file selective_global_re_in_se_de_0.2_Q2_1/train_mix --num_iter 5 --bsz 16

# # selective local, 4 times
# CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file selective_local_re_in_se_de_0.05_Q2_1/train_mix --num_iter 5 --bsz 16
# CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file selective_local_re_in_se_de_0.1_Q2_1/train_mix --num_iter 5 --bsz 16
# CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file selective_local_re_in_se_de_0.2_Q2_1/train_mix --num_iter 5 --bsz 16



####################-------------ablation study------------------###########################

# selective augmentation
CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file selective_local_re_in_se_de_0.1_Q2_1/train_de  --num_iter 5 --bsz 16 --comment de --group_head  
CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file selective_local_re_in_se_de_0.1_Q2_1/train_se  --num_iter 5 --bsz 16 --comment se
CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file selective_local_re_in_se_de_0.1_Q2_1/train_in  --num_iter 5 --bsz 16 --comment in
CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file selective_local_re_in_se_de_0.1_Q2_1/train_re  --num_iter 5 --bsz 16 --comment re


# random augmentation
CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file random_re_in_sw_de_0.1_1/train_de  --num_iter 5 --bsz 16 --comment de --group_head  
CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file random_re_in_sw_de_0.1_1/train_sw  --num_iter 5 --bsz 16 --comment sw
CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file random_re_in_sw_de_0.1_1/train_in  --num_iter 5 --bsz 16 --comment in
CUDA_VISIBLE_DEVICES=3 python clf.py --dataset bbc_50 --train_file random_re_in_sw_de_0.1_1/train_re  --num_iter 5 --bsz 16 --comment re


