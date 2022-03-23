## MovieLens

[MovieLens](https://grouplens.org/datasets/movielens/latest/)

## Criteo

[Criteo](https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310) 是经典的CTR预估比赛数据集，来源于[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/data) 比赛。

**Summary**

数据来自Criteo的流量数据，对正负样本按照不同比例采样，并按照时间顺序排序。

* `train.csv` Criteo 7天的部分流量，4000万
* `test.csv` train数据后一天的数据，600万
* `random_submission.csv` 提交样本数据格式

**Detail**

每行数据包含一下特征：

* `Label` 点击是1，否则是0
* `I1-I13` 13列 number feature，主要是计算特征
* `C1-C26` 26列 categorical feature，已经三列成32位

```
<Label> <I1> ... <I13> <C1> ... <C26>
```

## Avazu

[Avazu](https://www.kaggle.com/c/avazu-ctr-prediction/data) 数据集是Kaggle比赛的数据集。

**Summary**

数据来自的Avazu流量数据，并按照时间顺序排序。

* `train.csv` 10天点击数据，约4000万
* `test.csv` train数据后一天的数据，约460万
* `random_submission.csv` 提交样本数据格式

**Detail**

```id: ad identifier
click: 0/1 for non-click/click
hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
C1 -- anonymized categorical variable
banner_pos：广告位置
site_id
site_domain
site_category
app_id
app_domain
app_category
device_id
device_ip
device_model
device_type
device_conn_type
C14-C21 -- anonymized categorical variables
```

## TaoBao UserBehavior

[UserBehavior](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649) 是阿里巴巴提供的一个淘宝用户行为数据集，用于**隐式反馈推荐**问题的研究。

**Summary**

Time scope|Behavior count|User count|Item count|Category count|Size
-----|-----|-----|-----|-----|-----
2017.11.25 - 2017.12.03|100,150,807|987,994|4,162,024|9,439|3.67G

**Detail**

每行数据包括5个字段

列名称|说明
-----|-----
用户ID|整数类型，序列化后的用户ID
商品ID|整数类型，序列化后的商品ID
商品类目ID|整数类型，序列化后的商品所属类目ID
行为类型|字符串，枚举类型，'pv', 'buy', 'cart', 'fav'
时间戳|行为发生的时间戳

```
1,2268318,2520377,pv,1511544070
1,2333346,2520771,pv,1511561733
1,2576651,149192,pv,1511572885
```

## Taobao Ali_Display_Ad_Click

[Ali_Display_Ad_Click](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56&userId=1) 是从淘宝网站中随机抽样了114万用户8天内的广告展示/点击日志（2600万条记录）。

**Summary**

* `raw_sample` 原始的样本骨架：用户ID，广告ID，时间，资源位，是否点击
* `ad_feature`	广告的基本信息：广告ID，广告计划ID，类目ID，品牌ID
* `user_profile` 用户的基本信息：用户ID，年龄层，性别等
* `raw_behavior_log` 用户的行为日志：用户ID，行为类型，时间，商品类目ID，品牌ID

**Detail**

`原始样本骨架raw_sample`

> 14万用户8天内的广告展示/点击日志（2600万条记录）构成原始的样本骨架，7天的做训练样本（20170506-20170512），用第8天的做测试样本（20170513）

* user_id：脱敏过的用户ID
* adgroup_id：脱敏过的广告单元ID
* time_stamp：时间戳
* pid：资源位
* noclk：为1代表没有点击为0代表点击
* clk：为0代表没有点击为1代表点击

`广告基本信息表ad_feature`

> 一个广告ID对应一个商品（宝贝），一个宝贝属于一个类目，一个宝贝属于一个品牌

* adgroup_id：脱敏过的广告ID
* cate_id：脱敏过的商品类目ID
* campaign_id：脱敏过的广告计划ID
* customer_id:脱敏过的广告主ID
* brand：脱敏过的品牌ID
* price: 宝贝的价格

`用户基本信息表user_profile`
* userid：脱敏过的用户ID
* cms_segid：微群ID
* cms_group_id：cms_group_id
* final_gender_code：性别 1:男,2:女
* age_level：年龄层次
* pvalue_level：消费档次，1:低档，2:中档，3:高档
* shopping_level：购物深度，1:浅层用户,2:中度用户,3:深度用户
* occupation：是否大学生 ，1:是,0:否
* new_user_class_level：城市层级

`用户的行为日志behavior_log`

> 全部用户22天内的购物行为，共七亿条记录

* user：脱敏过的用户ID
* time_stamp：时间戳
* btag：行为类型,  ipv, cart, fav, buy
* cate：脱敏过的商品类目
* brand: 脱敏过的品牌词
