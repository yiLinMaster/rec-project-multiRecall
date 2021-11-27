## Project2-Multi-Recall实践

### 一、主要方法：

文章推荐场景下多路召回实践，使用user_baseCF，item_basedCF, matrix_basedCF, FM算法

recall策略
   * item_cf: 保存相似度最高的文章列表-算法：basic 相似度度量 + 关联关系：click_time权重，类别权重 

   * user_cf: 保存相似度最高的用户列表-算法：basic 相似度度量 + 关联关系：归一化的平均用户活跃度(这里有点问题？eg活跃度都高的人都看过不能更说明相似)

   * matrix_cf: 保存相似度最高的文章列表-根据matrix_cf计算的article_embedding用balltree索引树算法+欧式距离计算相似性获取最相近的向量

   * fm: 模型：fm, 损失策略：交叉熵， 优化方法:随机梯度下降
     
        * fm-i2i：保存相似度最高的文章列表-使用fm算法结果得到各文章隐向量再进行计算，同matrix_cf
        * fm-u2u：保存相似度最高的用户列表-使用fm算法结果得到各用户隐向量再进行计算
          
        

### 二、代码结构

#### 1.feature_server: 提取user_feature, item_feature存于redis db1，db2

#### 2.recall_server：调用各策略算法,保存对应中间结果在redis

    * item_cf: i2i_sim_list - db4
    * user_cf: u2u_sim_list - db5
    * matrix_cf: i2i_sim_list - db3
    * fm: i2i_sim_list - db6,  i_embedding - db7, u_feature_embedding - db8

#### 3.model:算法训练部分
    * data_processing: 构建正负样本，将样本特征hash化，将数据转化为tfrecords形式保存
    * inputs.py:class InputFn输入类--读取、解析tfrecords文件；dataset构建、分批+拉去ps中隐向量与x乘积；返回迭代器
    * fm.py:fm_fn--predict以及计算loss, setup_graph--SGD以及参数更新
    * mf.py:mf算法，同project1
    * train.py:定义训练过程，定义ps，inputfn,AUCUtil类，将数据iterator和setup_graph绑定,建立sess开始训练(以及其他交互，如将更新参数回传ps...)
    * ps.py
    * auc.py

#### 4.vector_server:定义向量服务部分VectorServer类，快速获取最相似向量

由于user_embedding中env等信息需要在引擎发出请求的时候获得，所以不能预先都存在redis里；需要在组装好user_embedding之后调用vector_server。所以预先封装一个可以提供vector_server的接口

#### 5.server_inf：主体部分:接收引擎请求，计算/取出各个策略的recall结果，汇总排序返回结果（定义RecallServer类进行统一调度，其中定义各feature_pool、中间结果pool等）

* def set_user_info(self, user_info):从召回引擎请求中得到初步user_info{"user_id":uid},并拉取feature_server中user_feature
* 调用各召回策略接口
  * item_cf：针对当前用户各个history记录找到相似文章列表再进行加权合并，返回最终排列截取结果
  * user_cf：遍历当前用户各个相似用户的history记录并集中的文章， 针对每一个item计算与目标用户历史记录中的items的相似度再加权累加排序
  * matrix_cf: 同item_cf
  * fm_i2i: 同item_cf
  * fm_u2i: 先实时调用feature_server完成user_embedding构造，再在线调用vector_server进行挑选最近item
* def merge_rec_res：将各路结果统一度量（归一化）之后可结合各路不同权值weight在排序返回item在各路计算上累加的最终结果

### 三、原始数据描述

    * click_log : user_id,article_id,timestamp,environment,region
    * articles : article_id,category_id,created_at_ts,words_count
    * matrixcf_articles_embedding: article_id, embedding



### 四、学习点小结：

* redis db的使用:pool.set/ pool.get/pool.keys/pool.pipeline

* tqdm进度条好看

* MinMaxScaler归一化

* np.linalg.norm归一化

* BallTree算法使用：neighbors.BallTree(), .query

* 正负样本随机取样

* 复习tfrecords存储和读取方法：get_tfrecords_example, _parse_example + interleave + map + batch + iterator,iterator.get_next

* 复习tf.data.Dataset构建：list_files指定文件路径,.repeat,.interleave,.map,.batch,.shuffle,.prefetch以及返回迭代器

* 交叉熵loss: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits = out_, labels = inputs["label"]))
    
* 复习模型定义过程:定义predict和计算loss的函数fm_fn + setup_graph-with tf.variable_scope("net_graph", reuse = is_test)进行梯度计算SGD和参数更新

* 复习模型训练过程：定义ps，inputfn类，将数据iterator和setup_graph绑定，建立sess开始训练(以及其他交互，如将更新参数回传ps...)

* 复习valid_step输出验证集上结果:try except

* 理解线上召回流程：召回服务统一调度模块定义recallserver类，进行各路召回结果进行再加权以及结果汇总。具体每路召回如fm_usi，必要信息如user_env等在引擎发出请求时实时传入，然后在线读取feature_server对应的redis组装成user_embedding,其它数据可以预先存于其它redis（如fm_i2i_recall中间结果等,不及时更新）,预先计划好；然后调用vector_server算出相似电影列表。

* item_cf 计算i2i_sim,user_cf计算u2u_sim之后结合user_hists的召回方法

* merge_rec_res中对各路返回结果进行归一化，并且将各路算法在某一item上的结果进行累加

    

### 五、疑问和修改尝试：
* 疑问1：user_cf中衡量用户相似度使用每对用户平均的归一化活跃度作为权重不一定起正面作用吧？

* 改动1：user_cf计算相似性中去掉平均活跃度因子，加入地区作为规则

* 疑问2：fm算法部分-直接对embedding（vi * xi）求导而不是对各个特征位对应的vi求导？

  **初步理解: 这里的fm实现是通过对于每个特征值训练出一个embedding vector隐向量vi+wi；可以理解为先对特征进行onehot编码，然后再为每个特征位训练一个隐向量。然后用二维特征lr公式预测相似度。所以也可以认为fm为每个特征值训练出一个隐向量，可用于特征值embedding**

  **而如果将fm用于特征值embedding，那么与matrix_based cf做embedding的区别？**

  **matrix_based cf：1）使用数据：只用评分矩阵， 2）训练目的：就是求user_embedding和item_embedding，使之乘积逼近评分值， 3）使用：user_embedding和item_embedding乘积预测评分；u2u，i2i使用欧式距离等进行相似度计算；u2i使用欧式距离进行相似度计算也可以理解（类lfm）**

  **fm：1）使用数据：利用user和item的很多其他特征，2）训练目的：为每个特征位训练隐向量weights（也可以理解为为各个特征值训练feature_embedding)，然后用包含二次特征项的LR逼近实际评分值， 3）使用：features+weights/直接feature_embedding放到二次项lr公式预测评分。如果要做embedding要先将各特征值对应的embedding加和（？这里的理解也有疑问，要参考论文是否可以这样用），then u2u，i2i使用欧式距离等进行相似度计算；u2i使用欧式距离衡量相似性（这个是否有意义还要看论文理解？）**

  疑问3：使用fm进行user_embedding、item_embedding直接将各特征值对应的embedding相加，怎么理解？--待看论文

  疑问4：使用mf/fm得到user_embedding、item_embedding之后，在vector_server中用balltree欧式距离排序是否有道理？为什么不按预测出的评分排序？且fm得到的embedding算欧式距离有意义？--待看论文

* 改动2：尝试修改input 特征为article_category_id,article_created_at_ts,article_words_count,user_environment,user_region--label，而不用user-id,item_id提高泛化性?--效果并不好，可能这些特征关联性不够。体验过程，效果只能留待以后提高

  TODO: timestamp和word_cnts没分桶直接用好像也有点不对

* 改动3：类似project1，训练完一遍train集合再输出test集--效果也并不明显

* 改动4：由于选用了多个item feature将item feature进行组合才能形成item embedding，对各feature_embedding进行了加和处理->input embedding

* 改动5：TODO：server_inf中created_time_weight可以放在item_cf_recall代码里，不占用在线调用时间资源

  

### 六、实验结果

* fm部分改动前:

  batch_size = 64
  weight_dim = 16
  learning_rate = 0.001

  ![1637564144051](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1637564144051.png)

* fm部分改动后：

  batch_size = 32
  weight_dim = 16
  learning_rate = 0.001

  test_show_step = 17850

  ![1637917435014](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1637917435014.png)