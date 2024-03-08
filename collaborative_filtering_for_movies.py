"""
 this is a item-based collaborative filtering system implementation.And this system implements main idea of item-based
 collaborative filtering thinking, in which we use similarity method to compute items' similarity and find
 recommendation items(or movies) for a target user.That means if items(in database table)' similarities with target
 user's history movies are greater than or equal to sim_threshold, system will recommend these items(or movies) to
 target user.Of course, recommended items(or movies) will be not in target user's history movies.

functions:
    connect_to_db: execute sql as a tool function (for developers)

    get_data: get data from table from database (for developers)

    create_ratings_matrix: create user(row)-item(colum) matrix (for developers)

    compute_similarity: compute items' similarity and output to a file (for developers)

    recommend: recommend for a user (both for developers and users)

    compute_similarity_using_for_user: get data, create rating matrix,
                                       and compute similarity, then output to a file (for users)

when use normally, just need 'compute_similarity_using_for_user', and 'recommend' functions

Created on May 24, 2023

@author: Dong Hao (3396363108@qq.com)
"""

import os
import time
from datetime import datetime as dt

import numpy as np
import numpy.linalg
import pandas as pd
import pymysql
import random


# pass successfully
def connect_to_db(user, pwd, database_name, sql, port=3306, host='localhost', charset='utf8'):
    """
    connect to database and execute custom sql(especially select type)
   :parameter:
        user: database user
        pwd: database password
        database_name: database name
        sql: select sql sentence
        port: connection port
        host: host name
        charset: encoding method
   :returns
        records: 2-dimension tuples of select result
    """

    conn = pymysql.connect(
        user=user,
        passwd=pwd,
        port=port,
        db=database_name,

        host=host,
        charset=charset
    )
    # get cursor
    cursor = conn.cursor()
    # define sql to execute, and this sql is function's parameter
    select_sql = sql
    cursor.execute(select_sql)
    # get data stored in records which is a two-dimension tuple
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return records


# pass successfully
def get_data(user, pwd, port, database_name='movies', table_name='ratings', host='localhost', charset='utf8', rate=1.0):
    """
    connect to database and select lines from which randomly with a changeable rate to return, the returns are:
        first: max user_id in database(!!!not in samples)
        second: max movie_id in database(!!!not in samples)
        third: a series of sampled tuples which are composed by lines of table in database

    here are some additional explanations:
        .tu_tms[0][0] -> max user_id in table
        tu_tms[0][1] -> max movie_id in table
    If you just use one return value,please use such as get_data(...)[0]

    :parameter:
        user: database user
        pwd: database password
        port: connection port
        database_name: database name
        table_name: table name in database
        host: host name
        charset: encoding method
        rate: sample rate from table in database
    :returns:
        max_user_id: max user id in database table
        max_movie_id: min user id in database table
        records: 2-dimension tuples of sampling result
    """

    # connect to database
    conn = pymysql.connect(
        user=user,
        passwd=pwd,
        port=port,
        db=database_name,

        host=host,
        charset=charset
    )
    # get cursor
    cursor = conn.cursor()
    # define sql to select * from destination table
    select_sql = 'select * from %s where rand() < %s' % (table_name, rate)
    cursor.execute(select_sql)
    # get data stored in records which is a two-dimension tuple
    records = cursor.fetchall()
    # this sql aims to get total number of user and movies
    get_total_num_sql = 'select max(userId + 0),max(movieId + 0) from %s' % table_name
    cursor.execute(get_total_num_sql)
    # tu_tms is total user(in fact the max user_id,and I just regard it as total users) and
    # total movies(the explanation is the same as above)
    tu_tms = cursor.fetchall()
    max_user_id = tu_tms[0][0]
    max_movie_id = tu_tms[0][1]
    conn.close()
    return max_user_id, max_movie_id, records


# pass successfully
def create_ratings_matrix(max_user_id, max_movie_id, records):
    """
    use max_user_id,max_movie_id and records to create information(user_item) matrix

    :parameter:
        max_user_id: max user id in matrix
        max_movie_id: min user id in matrix
        records: 2-dimension tuples, each of whose line  is a 'user-item' row
    :returns
        ratings_matrix: an information(user_item) matrix,like this(exclude head line):

                  |      1     2     3     4     5    6   movie_id
         _________|_____________________________________________________________________________________________________
               1  |     3.2   3.1  5.0
         _________|_____________________________________________________________________________________________________
               2  |           2.1               4.0
         _________|_____________________________________________________________________________________________________
               3  |
         _________|_____________________________________________________________________________________________________
          user_id |
                  |
    """

    # creat ratings_matrix (max_user_id * max_movie_id)
    ratings_matrix = np.mat(np.zeros((int(max_user_id), int(max_movie_id))))
    # add ratings into ratings_matrix.record is consist of userId(index is 0),movie_id(index is 1),
    # rating (index is 2) and timestamp (index is 3 but is not used).pay attention!!In order to correspond
    # with matrix index rules,I make user_id-1 and movie_id-1 to set rating in matrix,so if you need take
    # out a rating,you should use like this:
    # user_id:u,movie_id:m -> ratings_matrix[u-1][m-1]
    for record in records:
        user_id = int(record[0])
        movie_id = int(record[1])
        rating = float(record[2])
        ratings_matrix[user_id - 1, movie_id - 1] = rating
    # set rating 2.5 for zero ratings in ratings_matrix,you can try other methods
    ratings_matrix = np.where(ratings_matrix == 0, 2.5, ratings_matrix)
    # SVD decomposition begin-------------------------------------------------------------------------------------------
    # U, S, V = numpy.linalg.svd(ratings_matrix)
    # k = 0
    # sum = 0
    # count_sum = 0
    # for ele in numpy.nditer(S):
    #     sum += ele ** 2
    # for r in S:
    #     if count_sum / sum >= 0.9:
    #         break
    #     count_sum += r ** 2
    #     k += 1
    # low_dimension_matrix = V[:k, :]
    # return low_dimension_matrix
    # SVD end-----------------------------------------------------------------------------------------------------------
    return ratings_matrix


# SVD
def create_ratings_matrix_SVD(max_user_id, max_movie_id, records):
    """
    use max_user_id,max_movie_id and records to create information(user_item) matrix

    :parameter:
        max_user_id: max user id in matrix
        max_movie_id: min user id in matrix
        records: 2-dimension tuples, each of whose line  is a 'user-item' row
    :returns
        ratings_matrix: an information(user_item) matrix,like this(exclude head line):

                  |      1     2     3     4     5    6   movie_id
         _________|_____________________________________________________________________________________________________
               1  |     3.2   3.1  5.0
         _________|_____________________________________________________________________________________________________
               2  |           2.1               4.0
         _________|_____________________________________________________________________________________________________
               3  |
         _________|_____________________________________________________________________________________________________
          user_id |
                  |
    """

    # creat ratings_matrix (max_user_id * max_movie_id)
    ratings_matrix = np.mat(np.zeros((int(max_user_id), int(max_movie_id))))
    # add ratings into ratings_matrix.record is consist of userId(index is 0),movie_id(index is 1),
    # rating (index is 2) and timestamp (index is 3 but is not used).pay attention!!In order to correspond
    # with matrix index rules,I make user_id-1 and movie_id-1 to set rating in matrix,so if you need take
    # out a rating,you should use like this:
    # user_id:u,movie_id:m -> ratings_matrix[u-1][m-1]
    for record in records:
        user_id = int(record[0])
        movie_id = int(record[1])
        rating = float(record[2])
        ratings_matrix[user_id - 1, movie_id - 1] = rating
    # set rating 2.5 for zero ratings in ratings_matrix,you can try other methods
    ratings_matrix = np.where(ratings_matrix == 0, 2.5, ratings_matrix)
    # SVD decomposition begin-------------------------------------------------------------------------------------------
    U, S, V = numpy.linalg.svd(ratings_matrix)
    k = 0
    sum = 0
    count_sum = 0
    for ele in numpy.nditer(S):
        sum += ele
    for r in S:
        if count_sum / sum >= 0.99:
            break
        count_sum += r
        k += 1
    low_dimension_matrix = V[:k, :]
    return low_dimension_matrix
    # SVD end-----------------------------------------------------------------------------------------------------------


# SVD
def compute_similarity_SVD(information_matrix, method='adjusted_cos'):
    """
    receive a matrix and compute similarity of every two column,then store it in a 'similarity.csv'

    :parameter:
        information_matrix: 'user-item' form information matrix
        method: method of computing similarity(here just providing one:adjusted_cosine)
    :returns:
        path: relative path of file 'similarity.csv'
    """

    # create a dir named 'similarity_data',and create a file named 'similarity+time'
    folder = os.path.exists('similarity_data')
    if not folder:
        os.mkdir('./similarity_data')
    # create t_time to record time information  to use it as similarity file's name
    t_time = dt.strftime(dt.now(), '%Y%m%d-%H%M%S')
    # create similarity storing file
    path = './similarity_data/' + t_time + '.csv'
    similarity_store_file = open(path, 'xb')
    # computing similarity method one:adjusted cosine
    if method == 'adjusted_cos':
        information_matrix = np.array(information_matrix)
        row, col = np.shape(information_matrix)
        # buffer is to store computations of process in order to reduce writing to file's time,it stores a series of
        # cosine similarity of each outer iteration, and write to file at once
        buffer = ''
        # compute cosine similarity, item_static is former line's No. of each iteration and item_moving is the
        # latter line's No. of each iteration to the whole iteration, compute these two lines' cosine similarity
        for item_static in range(0, col):
            for item_moving in range(item_static + 1, col):
                # inner product of these two line
                inner_product = np.inner(information_matrix[:, item_static], information_matrix[:, item_moving])
                # 2-norms of these two line respectively
                norm_static = np.linalg.norm(information_matrix[:, item_static])
                norm_moving = np.linalg.norm(information_matrix[:, item_moving])
                # avoid zero in denominator, if so, set cosine similarity as zero
                if norm_static == 0 or norm_moving == 0:
                    buffer = buffer + str(0) + ','
                else:
                    cosine_sim = inner_product / (norm_static * norm_moving)
                    buffer = buffer + str(cosine_sim) + ','
            #   write cosine similarity to similarity_store_file using bytes(to facilitate removing final character of
            #   file) and clear buffer
            similarity_store_file.write((buffer[0: -1] + '\n').encode())
            buffer = ''
    # here you can define other similarity computing methods
    else:
        pass
    # remove final '\n' of file
    similarity_store_file.seek(-1, os.SEEK_END)
    similarity_store_file.truncate()
    # close file object
    similarity_store_file.close()
    # return the path of storing file
    return path


# pass successfully
def compute_similarity(information_matrix, method='adjusted_cos'):
    """
    receive a matrix and compute similarity of every two column,then store it in a 'similarity.csv'

    :parameter:
        information_matrix: 'user-item' form information matrix
        method: method of computing similarity(here just providing one:adjusted_cosine)
    :returns:
        path: relative path of file 'similarity.csv'
    """

    # create a dir named 'similarity_data',and create a file named 'similarity+time'
    folder = os.path.exists('similarity_data')
    if not folder:
        os.mkdir('./similarity_data')
    # create t_time to record time information  to use it as similarity file's name
    t_time = dt.strftime(dt.now(), '%Y%m%d-%H%M%S')
    # create similarity storing file
    path = './similarity_data/' + t_time + '.csv'
    similarity_store_file = open(path, 'xb')
    # computing similarity method one:adjusted cosine
    if method == 'adjusted_cos':
        # subtract means of users' ratings, here fist transfer means and information_matrix to matrix to implement
        # column-based-broadcasting, then transfer changed information_matrix to n-array to facilitate later computing
        means = np.mat(np.mean(information_matrix, axis=1))
        means = means.T
        information_matrix = information_matrix - means
        information_matrix = np.array(information_matrix)
        row, col = np.shape(information_matrix)
        # buffer is to store computations of process in order to reduce writing to file's time,it stores a series of
        # cosine similarity of each outer iteration, and write to file at once
        buffer = ''
        # compute cosine similarity, item_static is former line's No. of each iteration and item_moving is the
        # latter line's No. of each iteration to the whole iteration, compute these two lines' cosine similarity
        for item_static in range(0, col):
            for item_moving in range(item_static + 1, col):
                # inner product of these two line
                inner_product = np.inner(information_matrix[:, item_static], information_matrix[:, item_moving])
                # 2-norms of these two line respectively
                norm_static = np.linalg.norm(information_matrix[:, item_static])
                norm_moving = np.linalg.norm(information_matrix[:, item_moving])
                # avoid zero in denominator, if so, set cosine similarity as zero
                if norm_static == 0 or norm_moving == 0:
                    buffer = buffer + str(0) + ','
                else:
                    cosine_sim = inner_product / (norm_static * norm_moving)
                    buffer = buffer + str(cosine_sim) + ','
            #   write cosine similarity to similarity_store_file using bytes(to facilitate removing final character of
            #   file) and clear buffer
            similarity_store_file.write((buffer[0: -1] + '\n').encode())
            buffer = ''
    # here you can define other similarity computing methods
    else:
        pass
    # remove final '\n' of file
    similarity_store_file.seek(-1, os.SEEK_END)
    similarity_store_file.truncate()
    # close file object
    similarity_store_file.close()
    # return the path of storing file
    return path


# pass successfully
def recommend(user_id, start_time, end_time, sim_threshold, sim_location, top_n, table_name=''):
    """
    collect recent users' behaviors,and then find all items based on users' behaviors whose similarity are greater t-
    han or equal to sim_threshold from input location.Finally, remove items already behaved in recommend list.

    :parameter:
        user_id: target user id for recommending
        start_time: start time of target user's movie id history, such as '1995-01-09 00:00:00'
        end_time: end time of target user's movie id history, such as '2019-11-21 00:00:00'
        sim_threshold: threshold of similarity
        sim_location: relative path of similarity.csv
        top_n: top N recommended items
        table_name : table name of database
    :returns:
        recommending_list:recommending result
    """

    # convert start_time and end_time to timestamp
    st_time_struct = time.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    st_time_stamp = time.mktime(st_time_struct)
    ed_time_struct = time.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    ed_time_stamp = time.mktime(ed_time_struct)
    # connect to database and get past movieIds of user_id in table during start_time and end_time
    sql = f'select movieId from {table_name} where' \
          f' (userId + 0) = {user_id}' \
          f' and ((timestamp + 0) between {st_time_stamp} and {ed_time_stamp})'
    history_tuple = connect_to_db('root', 'root', 'movies', sql)
    # convert to array to use flatten method, and flatten history_tuple to history_list and remove duplicated elements
    history_list = np.array(history_tuple, int).flatten('F')
    history_list = list(set(history_list))
    # find movieIds in sim_location whose similarity with movieIds above are greater than or equal to sim_threshold
    # read similarity file
    sim = pd.read_csv(sim_location, header=None, dtype=float, engine='python')
    # create recommending list
    recommending_list = []
    # find and record
    '''
    iterate iterator sim to travel every element of it,every element is associated with
    <index+1, index+1+row_i+1> presenting similarity of these two movies' similarity,
    whether index+1 or index+1+row_i+1 is in history_list, we can parse the other part of this <key,value> form
    and decide whether to add its movie id to recommending_list according to sim_threshold (understanding way 1)
    '''
    for index, row in sim.iterrows():
        for row_i in range(len(row)):
            # skip this loop once encounter nan value
            if np.isnan(row[row_i]):
                break
            """former movies' parse base on travelling item (understanding way 2)"""
            # if the movie_id(row_i in matrix,actually index + 1 + row_i + 1 in reality) is in user's past history list,
            # and one of its former movies(index in matrix, actually index + 1 in reality)' similarity with which is
            # greater than or equal to the sim_threshold, add this former movie's id to recommending_list
            if (index + 1 + row_i + 1 in history_list) and (row[row_i] >= sim_threshold):
                recommending_list.append(index + 1)
            """latter movies' parse base on travelling item (understanding way 2)"""
            # if the movie_id(index in matrix, actually index + 1 in reality) is in user's past history list, and one
            # of its latter movies(row_i index + 1 + row_i + 1)'s similarity with which is greater than or equal to
            # the sim_threshold, add this latter movie's id to recommending_list
            if (index + 1 in history_list) and (row[row_i] >= sim_threshold):
                recommending_list.append(index + 1 + row_i + 1)
    recommending_list = list(set(recommending_list))
    # remove duplicated movieIds from recommending list
    recommending_list = list(set(recommending_list).difference(set(history_list)))
    # choose top n items to recommend
    if len(recommending_list) >= top_n:
        recommending_list = random.sample(recommending_list, top_n)
    # return recommending list
    return recommending_list


def compute_similarity_using_for_user(user, pwd, port, database_name='movies', table_name='ratings', host='localhost',
                                      charset='utf8', rate=1.0, sim_method='adjusted_cos'):
    """
    compute similarity and return a path of similarity information

    :param user: database user
    :param pwd: database password
    :param port: connection port
    :param database_name: database name
    :param table_name: table name whose similarity you want to compute
    :param host: host name
    :param charset: encoding charset
    :param rate: sampling ratio based on param table_name
    :param sim_method: method to compute similarity
    :returns:
        path of '*.csv' containing similarity information
    """

    u, m, records = get_data(user, pwd, port, database_name, table_name, host, charset, rate)
    matrix = create_ratings_matrix(u, m, records)
    path = compute_similarity(matrix, sim_method)
    return path


# SVD
def compute_similarity_using_for_user_SVD(user, pwd, port, database_name='movies', table_name='ratings',
                                          host='localhost', charset='utf8', rate=1.0, sim_method='adjusted_cos'):
    """
    compute similarity and return a path of similarity information

    :param user: database user
    :param pwd: database password
    :param port: connection port
    :param database_name: database name
    :param table_name: table name whose similarity you want to compute
    :param host: host name
    :param charset: encoding charset
    :param rate: sampling ratio based on param table_name
    :param sim_method: method to compute similarity
    :returns:
        path of '*.csv' containing similarity information
    """

    u, m, records = get_data(user, pwd, port, database_name, table_name, host, charset, rate)
    matrix = create_ratings_matrix_SVD(u, m, records)
    path = compute_similarity_SVD(matrix, sim_method)
    return path


if __name__ == '__main__':
    """ this is an example"""

    # compute similarity
    sim_path = compute_similarity_using_for_user_SVD('root', 'root', 3306, 'movies', 'ml100k', 'localhost', 'utf8',
                                                     1, 'adjusted_cos')
    print(sim_path)

    # recommend
    recommending_list = recommend(5, '1995-01-09 00:00:00', '2019-11-21 00:00:00', 0.8, sim_path, 10000, 'ml100k')
    print(recommending_list)
