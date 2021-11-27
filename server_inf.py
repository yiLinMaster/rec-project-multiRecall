# -*- coding: utf-8 -*-

from numpy.core.records import recarray
from numpy.lib.histograms import _unsigned_subtract
import redis
from vector_server import vector_gen
import json
import numpy as np
# 召回部分主体:接收引擎请求，计算/取出各个策略的recall结果，汇总排序返回结果
class RecallServer:
    def __init__(self):

        self.redis_url = "redis://@127.0.0.1:6379/"
        self.user_feature_pool = redis.from_url(self.redis_url + "1")
        self.item_feature_pool = redis.from_url(self.redis_url + "2")
        self.matrixcf_i2i_pool = redis.from_url(self.redis_url + "3")
        self.itemcf_i2i_pool = redis.from_url(self.redis_url + "4")
        self.usercf_u2u_pool = redis.from_url(self.redis_url + "5")
        self.fm_i2i_pool = redis.from_url(self.redis_url + "6")
        self.fm_item_embedding_pool = redis.from_url(self.redis_url + "7")
        self.fm_user_feature_embedding_pool = redis.from_url(self.redis_url + "8")

        # 定义缓存区
        self.user_info = {}
        self.item_info = {}
        self.current_user_feature = {}

        # 由于user_embedding实时构建，所以fm_u2i_recall需要定义vector_server
        print("Start vector server...")
        self.vector_server = vector_gen.VectorServer(self.fm_item_embedding_pool) 

    # step1：从召回引擎请求中得到初步user_info{"user_id":uid},并拉取feature_server中user_feature
    def set_user_info(self, user_info):
        u = user_info['user_id']
        self.current_user_feature = self.user_info.get(str(u),None)
        if self.current_user_feature is None:
            self.current_user_feature = json.loads(self.user_feature_pool.get(str(u)))
            self.user_info[str(u)] = self.current_user_feature

        print("Get rec res for user: %s"% str(u))
        # print("Current user feature include:")
        # for k, v in self.current_user_feature.items():
        #     print(k,":",v)
        # print("=" * 80)

    # step2: 调用各召回策略接口
    # 2.1 item_cf
    # 使用策略:针对当前用户各个history记录找到相似文章列表再进行加权合并，返回最终排列截取结果
    def get_item_cf_rec_res(self, recall_num = 30):    
        if len(self.current_user_feature) <= 0:
            print("未接收到当前用户信息")
            return 

        item_rank = {}
        hists = self.current_user_feature['hists']
        for loc, (i,_) in enumerate(hists):
            item_sim_ = json.loads(self.itemcf_i2i_pool.get(str(i)))

            # 针对具体用户行为再加权
            # note：这里通过引入再hists中的位置计算时序性！
            # loc越大loc_weight越大，即新文章权重越大
            loc_weight = (0.9**(len(hists) - loc))
        
            for j, wij in item_sim_:
                # 频控服务筛掉用户已经有过行为的item
                if j in hists:
                    continue
                
                # 理解：这里是在使用共现矩阵计算文章相似度的基础上再结合具体用户记录/补充一些权重
                item_i_info = self.item_info.get(i,None)
                if item_i_info is None:
                    item_i_info = json.loads(self.item_feature_pool.get(str(i)))
                    self.item_info[i] = item_i_info
                item_j_info = self.item_info.get(j,None)
                if item_j_info is None:
                    item_j_info = json.loads(self.item_feature_pool.get(str(j)))
                    self.item_info[j] = item_j_info

                # 改动：存入redis的时候已经用type_weight了，省去
                # 规则：created_time_weight
                # TODO：created_time_weight可以放在item_cf_recall代码里，不占用在线调用时间资源
                created_time_weight = np.exp(
                    0.7 ** np.abs(item_i_info['created_at_ts'] - item_j_info['created_at_ts'])
                )

                item_rank.setdefault(j,0)
                item_rank[j] += loc_weight * created_time_weight * wij
                
        item_rank= sorted(item_rank.items(), key=lambda x:x[1] ,reverse = True)
        item_rank = item_rank[:recall_num]
        print("=" * 80)
        print("recall_res -- item_cf_i2i:size == {}".format(len(item_rank)))
        print("=" * 80)

        return item_rank
    
    # 2.2 user_cf
    # 使用策略:遍历当前用户各个相似用户的history记录并集中的文章，
    # 针对每一个item计算与目标用户历史记录中的items的相似度再加权累加排序
    def get_user_cf_rec_res(self, recall_num = 30):
        if len(self.current_user_feature) <= 0:
            print("未接收到当前用户信息")
            return 

        u = self.current_user_feature['user_id']
        u_hists = self.current_user_feature['hists']

        u2u_sim = json.loads(self.usercf_u2u_pool.get(str(u)))

        item_rank = {}

        for v,wuv in u2u_sim:
            v_feature = self.user_info.get(str(v),None)
            if v_feature is None:
                v_feature = json.loads(self.user_feature_pool.get(str(v)))
                self.user_info[str(v)] = v_feature
            
            for i,_ in v_feature['hists']:
                # 频控
                if i in u_hists:
                    continue

                item_i_info = self.item_info.get(i,None)
                if item_i_info is None:
                    item_i_info = json.loads(self.item_feature_pool.get(str(i)))
                    self.item_info[i] = item_i_info

                item_rank.setdefault(i,0)

                for loc, (j, click_time) in enumerate(u_hists):
                    item_j_info = self.item_info.get(j,None)
                    if item_j_info is None:
                        item_j_info = json.loads(self.item_feature_pool.get(str(j)))
                        self.item_info[j] = item_j_info

                    type_weight = 1.0 if item_i_info['category_id'] == \
                        item_j_info['category_id'] else 0.7
                    
                    created_time_weight = np.exp(
                        0.7 ** np.abs(item_i_info['created_at_ts'] - item_j_info['created_at_ts'])
                    )
                    
                    loc_weight = (0.9**(len(u_hists) - loc))

                    item_rank[i] += loc_weight * created_time_weight * type_weight * wuv
                
        item_rank= sorted(item_rank.items(), key=lambda x:x[1] ,reverse = True)
        item_rank = item_rank[:recall_num]
        print("=" * 80)
        print("recall_res -- user_cf_u2u: size == {}".format(len(item_rank)))
        print("=" * 80)

        return item_rank

    # 2.3 matrix_cf : i2i
    # 使用策略同2.1
    def get_matrix_cf_rec_res(self, recall_num = 30):
        if len(self.current_user_feature) <= 0:
            print("未接收到当前用户信息")
            return 
        item_rank = {}
        hists = self.current_user_feature['hists']
        for loc, (i,_) in enumerate(hists):
            item_sim_ = json.loads(self.matrixcf_i2i_pool.get(str(i)))
            loc_weight = (0.9**(len(hists) - loc))
        
            for j, wij in item_sim_:
                # 频控服务筛掉用户已经有过行为的item
                if j in hists:
                    continue
                
                # 理解：这里是在使用共现矩阵计算文章相似度的基础上再结合具体用户记录/补充一些权重
                item_i_info = self.item_info.get(i,None)
                if item_i_info is None:
                    item_i_info = json.loads(self.item_feature_pool.get(str(i)))
                    self.item_info[i] = item_i_info
                item_j_info = self.item_info.get(j,None)
                if item_j_info is None:
                    item_j_info = json.loads(self.item_feature_pool.get(str(j)))
                    self.item_info[j] = item_j_info

                # 规则：type_weight， 同类型权重大
                type_weight = 1.0 if item_i_info['category_id'] == \
                    item_j_info['category_id'] else 0.7

                # 规则：created_time_weight
                created_time_weight = np.exp(
                    0.7 ** np.abs(item_i_info['created_at_ts'] - item_j_info['created_at_ts'])
                )

                item_rank.setdefault(j,0)
                item_rank[j] += loc_weight * created_time_weight *type_weight* wij
                
        item_rank= sorted(item_rank.items(), key=lambda x:x[1] ,reverse = True)
        item_rank = item_rank[:recall_num]
        print("=" * 80)
        print("recall_res -- matrix_cf_i2i: size == {}".format(len(item_rank)))
        print("=" * 80)

        return item_rank

    # 2.4 fm:i2i
    # 使用策略同上
    def get_fm_i2i_rec_res(self, recall_num = 30):
        if len(self.current_user_feature) <= 0:
            print("未接收到当前用户信息")
            return 
        item_rank = {}
        hists = self.current_user_feature['hists']
        for loc, (i,_) in enumerate(hists):
            item_sim_ = json.loads(self.fm_i2i_pool.get(str(i)))

            loc_weight = (0.9**(len(hists) - loc))
        
            for j, wij in item_sim_:
                # 频控服务筛掉用户已经有过行为的item
                if j in hists:
                    continue
                
                # 理解：这里是在使用共现矩阵计算文章相似度的基础上再结合具体用户记录/补充一些权重
                item_i_info = self.item_info.get(i,None)
                if item_i_info is None:
                    item_i_info = json.loads(self.item_feature_pool.get(str(i)))
                    self.item_info[i] = item_i_info
                item_j_info = self.item_info.get(j,None)
                if item_j_info is None:
                    item_j_info = json.loads(self.item_feature_pool.get(str(j)))
                    self.item_info[j] = item_j_info

                # 规则：type_weight， 同类型权重大
                type_weight = 1.0 if item_i_info['category_id'] == \
                    item_j_info['category_id'] else 0.7

                # 规则：created_time_weight
                created_time_weight = np.exp(
                    0.7 ** np.abs(item_i_info['created_at_ts'] - item_j_info['created_at_ts'])
                )

                item_rank.setdefault(j,0)
                item_rank[j] += loc_weight * created_time_weight *type_weight* wij
                
        item_rank= sorted(item_rank.items(), key=lambda x:x[1] ,reverse = True)
        item_rank = item_rank[:recall_num]
        print("=" * 80)
        print("recall_res -- fm_i2i: size == {}".format(len(item_rank)))
        print("=" * 80)

        return item_rank

    # 2.5fm: u2i
    # 实时构建user_embedding:从feature_server的redis里面读，不能全看fm recall_server里面的预计算值
    # 在线调用向量服务vector_server获取相似的item_embedding
    def get_fm_u2i_rec_res(self,recall_num = 30):
        if len(self.current_user_feature) <= 0:
            print("未接收到当前用户信息")
            return 
        user_environment = self.current_user_feature['environment']
        user_region = self.current_user_feature['region']
        user_environment_id = "envs_id="+str(user_environment) 
        user_region_id = "regions_id="+str(user_region) 
        emb = self.fm_user_feature_embedding_pool.mget([user_environment_id,user_region_id])
        # 勿忘json.loads
        emb = [json.loads(_) for _ in emb]
        # 构造user_embedding : user_environment,user_region
        # print(emb)
        emb = np.sum(np.asarray(emb), axis = 0,keepdims = True)
        # print(emb)
        # vector_server
        item_rank = self.vector_server.get_sim_item(emb, recall_num)
        # 注意这个函数设计的是返回一个列表item的结果，所以要取第一个结果
        item_rank = item_rank[0]
        item_rank = item_rank[:recall_num]
        print("=" * 80)
        print("recall_res -- fm_u2i: size == {}".format(len(item_rank)))
        print("=" * 80)
        return item_rank

    # step3:将各路结果统一度量（归一化）之后可结合各路不同权值weight在排序返回最终结果
    def merge_rec_res(self,item_ranks) :
        item_rec = {}
        # 对各路赋予一个权重
        for item_rank, weight in item_ranks:
            tmp = [_[1] for _ in item_rank]
            max_value = max(tmp)
            min_value = min(tmp)
            for i, w in item_rank:
                item_rec.setdefault(i,w)
                # 注意这里是对各路策略计算结果累加！
                item_rec[i] += weight * (w - min_value) / (max_value - min_value)
        
        print("=" * 80)
        print("all recall_res -- size == {}".format(len(item_rec)))
        print("=" * 80)
        
        return item_rec
       
if __name__ =='__main__':
    print("Start recall server...")
    rs = RecallServer()

    # 模拟来自请求的user_info
    rec_user = {'user_id':190000}

    rs.set_user_info(rec_user)
    item_cf_item_rank = rs.get_item_cf_rec_res()
    user_cf_item_rank = rs.get_user_cf_rec_res()
    matrixcf_item_rank = rs.get_matrix_cf_rec_res()
    fm_i2i_item_rank = rs.get_fm_i2i_rec_res()
    fm_u2i_item_rank = rs.get_fm_u2i_rec_res()
    
    item_rec = rs.merge_rec_res([(item_cf_item_rank, 1.0),
        (user_cf_item_rank, 1.0),(matrixcf_item_rank, 1.0),
        (fm_i2i_item_rank, 1.0),(fm_u2i_item_rank, 1.0)])

    print("=" * 80)
    print("current_user: {}, rec items: {}".format(rec_user['user_id'],item_rec))