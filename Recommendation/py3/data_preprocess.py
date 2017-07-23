# coding: utf-8
#
# data_preprocess.py
#
# Author: Huang Anbu
# Date: 2017.4
#
# Description: data processing, after execute this script, you will get new file "data.pkl", it
#   contains the following data structure:
#
#   - min_user_id
#   - max_user_id
#   - min_movie_id
#   - max_movie_id
#   - train_set
#
# Copyright©2017. All Rights Reserved. 
# ===============================================================================================


from basiclib import *

if __name__ == "__main__":
	cur_dir = os.getcwd()
	path = os.path.join(cur_dir, "dataset")
	
	os.chdir(path)
	data = {}
	max_user_id, min_user_id = 0, 1000000
	max_movie_id, min_movie_id = 0, 1000000
	with open("ratings.dat", "r", encoding='utf-8') as fin:
		for line in fin:
			ls = list(map(lambda x:int(x), line.split("::")))
			user, movie, rate = ls[0], ls[1], ls[2]
			max_user_id = max(max_user_id, user)
			min_user_id = min(min_user_id, user)
			
			max_movie_id = max(max_movie_id, movie)
			min_movie_id = min(min_movie_id, movie)
			
			if user not in data:
				data[user] = [(movie, rate)]
			else:
				data[user].append((movie, rate))
	
	if min_user_id == 1:
		max_user_id = max_user_id - 1
		min_user_id = min_user_id - 1
	
	if min_movie_id == 1:
		max_movie_id = max_movie_id - 1
		min_movie_id = min_movie_id - 1
		
	train_set = numpy.zeros((max_user_id+1, max_movie_id+1))
	
	os.chdir(cur_dir)
	for k, v in data.items():
		for m, r in v:
			train_set[k-1][m-1]=r 
			
	with open("data.pkl", "wb") as fout:
		pickle.dump((min_user_id, max_user_id, min_movie_id, max_movie_id, train_set), fout)
		