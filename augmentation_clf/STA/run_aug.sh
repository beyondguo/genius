
# #=================sst2new_50===========================
# python run_eda.py --dataset sst2new_50 --lang en --methods re,in,sw,de --p 0.1

# # # 不要swap，换成positive selection
# python run_sta.py --dataset sst2new_50 --lang en --methods re,in,se,de --bar Q2 --n_aug 1 --p 0.05
# python run_sta.py --dataset sst2new_50 --lang en --methods re,in,se,de --bar Q2 --n_aug 1 --p 0.1
# python run_sta.py --dataset sst2new_50 --lang en --methods re,in,se,de --bar Q2 --n_aug 1 --p 0.2

# python run_sta.py --dataset sst2new_50 --lang en --methods re,in,se,de --bar Q2 --n_aug 1 --p 0.05 --strategy global
# python run_sta.py --dataset sst2new_50 --lang en --methods re,in,se,de --bar Q2 --n_aug 1 --p 0.1 --strategy global
# python run_sta.py --dataset sst2new_50 --lang en --methods re,in,se,de --bar Q2 --n_aug 1 --p 0.2 --strategy global



#------for wordnet-eda--------
python run_eda.py --dataset bbc_50 --lang en --methods re,in,sw,de --p 0.1
python run_eda.py --dataset bbc_100 --lang en --methods re,in,sw,de --p 0.1

python run_eda.py --dataset ng_50 --lang en --methods re,in,sw,de --p 0.1
python run_eda.py --dataset ng_100 --lang en --methods re,in,sw,de --p 0.1

python run_eda.py --dataset imdb_50 --lang en --methods re,in,sw,de --p 0.1
python run_eda.py --dataset imdb_100 --lang en --methods re,in,sw,de --p 0.1

python run_eda.py --dataset yahoo_50 --lang en --methods re,in,sw,de --p 0.1
python run_eda.py --dataset yahoo_100 --lang en --methods re,in,sw,de --p 0.1


