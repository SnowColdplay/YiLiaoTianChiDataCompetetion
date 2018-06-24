import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import re
import warnings
warnings.filterwarnings("ignore")
import time

# 把数值形的都横着放
# 根据四段年龄细化特征
# 做symbol 阴性阳性特征

print('只包括数字的')


def is_number(uchar):
    try:
        float(uchar)
        return 1
    except:
        return 0


def convert_float(number):
    try:
        return float(number)
    except:
        return -999


def conver_label(number):
    try:
        return float(number)
    except:
        return -999


def pre_process(train, test):
    lbl = preprocessing.LabelEncoder()
    for col in ['vid']:
        train[str(col) + 'lbl'] = lbl.fit_transform(train[col])
        test[str(col) + 'lbl'] = lbl.fit_transform(test[col])

    # train['field_results'] = train['field_results'].apply(convert_float)
    # test['field_results'] = test['field_results'].apply(convert_float)

    # train=train[train['field_results']!=-999]
    # test = test[test['field_results'] != -999]

    for lie in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']:
        del test[lie]

    return train, test


def quchong(data):
    print('每个vid总共的检查项目')
    col = [c for c in data if c not in ['field_results']]
    data1 = data[col]
    data1 = data1.drop_duplicates()
    return data1

def caichao(data):
    df1 = data.copy()
    # df1 = data[data['table_id'] == '0102']
    df1['肝'] = df1['field_results'].apply((lambda x: 1 if str(x).__contains__('肝') else 0))
    df1 = df1[df1['肝'] == 1]
    df1 = df1.groupby(['vid'], as_index=False)['field_results'].agg({'r1': np.sum})
    df1['is_肝1'] = df1['r1'].apply(lambda x: 1 if str(x).__contains__('脂肪肝') else 0)
    df1['is_肝'] = df1['r1'].apply(
        lambda x: 1 if str(x).__contains__('脂肪肝（轻度）') else 2 if str(x).__contains__('脂肪肝（中度）') else 3
        if str(x).__contains__('脂肪肝（重度）') else 4 if str(x).__contains__('脂肪肝（非均匀性）') else 5 if str(x).__contains__(
            '脂肪肝趋势') else 0)
    df1 = df1[['vid', 'is_肝', 'is_肝1']]
    data = pd.merge(data, df1, on='vid', how='left')
    data['is_肝'] = data['is_肝'].fillna(-1)
    data['is_肝1'] = data['is_肝1'].fillna(-1)
    del df1

    df1 = data.copy()
    df1['胆'] = df1['field_results'].apply((lambda x: 1 if str(x).__contains__('胆') else 0))
    df1 = df1[df1['胆'] == 1]
    df1 = df1.groupby(['vid'], as_index=False)['field_results'].agg({'r1': np.sum})
    df1['is_胆'] = df1['r1'].apply(lambda x: 1 if str(x).__contains__('胆固醇') else 2 if str(x).__contains__('胆囊') else 0)
    df1 = df1[['vid', 'is_胆']]
    data = pd.merge(data, df1, on='vid', how='left')
    del df1

    df1 = data.copy()
    df1['肾'] = df1['field_results'].apply((lambda x: 1 if str(x).__contains__('肾') else 0))
    df1 = df1[df1['肾'] == 1]
    df1 = df1.groupby(['vid'], as_index=False)['field_results'].agg({'r1': np.sum})
    df1['is_肾结石'] = df1['r1'].apply(lambda x: 1 if (str(x).__contains__('肾结')) | (str(x).__contains__('肾囊')) else 0)
    df1 = df1[['vid', 'is_肾结石']]
    data = pd.merge(data, df1, on='vid', how='left')
    del df1

    df1 = data.copy()
    df1['甲状腺'] = df1['field_results'].apply((lambda x: 1 if str(x).__contains__('甲状腺') else 0))
    df1 = df1[df1['甲状腺'] == 1]
    df1 = df1.groupby(['vid'], as_index=False)['field_results'].agg({'r1': np.sum})
    df1['is_甲状腺彩超'] = df1['r1'].apply(lambda x: 1 if (str(x).__contains__('明显异常')) else 0)
    df1 = df1[['vid', 'is_甲状腺彩超']]
    data = pd.merge(data, df1, on='vid', how='left')
    del df1

    # "-------------------------------one hot--------------------------------------------"
    # dummies = pd.get_dummies(data[['is_肝', 'is_胆', 'is_肾结石', 'is_甲状腺彩超']],
    #                          columns=['is_肝', 'is_胆', 'is_肾结石', 'is_甲状腺彩超'])
    # data = pd.concat([data, dummies], axis=1)

    return data


def jiankang(data):
    tmp = data[data['table_id'] == '2302']
    tmp1 = tmp.groupby(['vid'], as_index=False)['field_results'].agg({'r1': np.sum})
    tmp1['is_jiankang'] = tmp1['r1'].apply(
        lambda x: 1 if str(x).__contains__('健康') else 2 if str(x).__contains__('亚健康') else 3 if str(x).__contains__(
            '肥健康') else 0)
    del tmp1['r1']
    data = pd.merge(data, tmp1, on='vid', how='left')

    # "-------------------------------one hot--------------------------------------------"
    # dummies = pd.get_dummies(data[['is_jiankang']], columns=['is_jiankang'])
    # data = pd.concat([data, dummies], axis=1)
    # del data['is_jiankang']

    return data


def sex(data):
    # 0121是妇科检查，只要包含0121就是女性 ，0120是男性。分为年轻女性，老年女性，年轻男性，老年男性

    df1 = data[data['table_id'] == '0121']
    df1 = df1.groupby(['vid'], as_index=False)['field_results'].agg({'r1': np.sum})
    df1['年龄四段'] = df1['r1'].apply(lambda x: 1 if (str(x).__contains__('绝经')) else 2)
    df1 = df1[['vid', '年龄四段']]

    df2 = data[data['table_id'] == '0120']
    df2 = df2.groupby(['vid'], as_index=False)['field_results'].agg({'r1': np.sum})
    df2['年龄四段'] = df2['r1'].apply(
        lambda x: 3 if (str(x).__contains__('正常')) | (str(x).__contains__('未见异常')) | (str(x).__contains__('未见明显异常'))
                       | (str(x).__contains__('大小形态尚规则')) else 4)
    df2 = df2[['vid', '年龄四段']]

    df3 = pd.concat([df1, df2])
    data = pd.merge(data, df3, on='vid', how='left')

    # 给ibm分段时会用到这个
    tmp = data[data['table_id'] == '0121']
    tmp = tmp[['vid', 'table_id']]
    tmp = tmp.rename(columns={'table_id': 'is_guniang'})
    data = pd.merge(data, tmp, on='vid', how='left')
    data['is_guniang'] = data['is_guniang'].fillna(-1)
    data['is_guniang'] = data['is_guniang'].apply(lambda x: 1 if x == '0121' else -1)

    # "-------------------------------one hot--------------------------------------------"
    # dummies = pd.get_dummies(data[['年龄四段', 'is_guniang']], columns=['年龄四段', 'is_guniang'])
    # data = pd.concat([data, dummies], axis=1)


    return data


def neike(data):
    # 0409是内科，包含病史，糖尿病等
    df1 = data[data['table_id'] == '0409']
    df1 = df1.groupby(['vid'], as_index=False)['field_results'].agg({'r1': np.sum})
    df1['is_neike_yichang'] = df1['r1'].apply(
        lambda x: 1 if (str(x).__contains__('未发现')) | (str(x).__contains__('未见')) else 0)  # 内科没发现异常是1
    df1 = df1[['vid', 'is_neike_yichang', 'r1']]
    data = pd.merge(data, df1, on='vid', how='left')

    # "-------------------------------one hot--------------------------------------------"
    # dummies = pd.get_dummies(data[['is_neike_yichang']], columns=['is_neike_yichang'])
    # data = pd.concat([data, dummies], axis=1)
    # del data['is_neike_yichang']

    del data['r1']
    "---术后-----"
    tmp = df1[df1['is_neike_yichang'] == 0]  # 内科异常的
    tmp = tmp[['vid', 'r1']]
    tmp['is_shuhou'] = tmp['r1'].apply(lambda x: 1 if str(x).__contains__('术后') else 0)

    "--------------------------------------------------高血压，糖尿病等-------------------------------------------------"
    for col in ['糖尿病史', '高血压史']:
        tmp['is_' + str(col)] = tmp['r1'].apply(lambda x: 1 if str(x).__contains__(col) else 0)
        tmp['is_' + str(col) + '_治疗'] = tmp['r1'].apply(
            lambda x: 1 if (str(x).__contains__(col)) & (str(x).__contains__('治疗中')) else 2
            if (str(x).__contains__(col)) & (str(x).__contains__('未治疗')) else 3
            if (str(x).__contains__(col)) & (str(x).__contains__('间断治疗'))
            else 0)

    for col in ['冠心病史', '脂肪肝史', '脑', '胃', '甲状腺', '肾']:
        tmp['is_' + str(col)] = tmp['r1'].apply(lambda x: 1 if str(x).__contains__(col) else 0)
    # "------------------------------------------------同时有多个病史----------------------------------------------"
    # #同时有两个病史
    col = ['糖尿病史', '高血压史']
    tmp['is_' + str(col[0]) + str(col[1])] = tmp['r1'].apply(
        lambda x: 1 if (str(x).__contains__(col[0])) & (str(x).__contains__(col[1]))else 0)

    "未提取的特征：血糖，血脂，只有600多条有"

    del tmp['r1']
    data = pd.merge(data, tmp, on='vid', how='left')

    # "-------------------------------one hot--------------------------------------------"
    # c = []
    # c.append('is_shuhou')
    # for col in ['糖尿病史', '高血压史']:
    #     c.append('is_' + str(col))
    #     c.append('is_' + str(col) + '_治疗')
    # for col in ['冠心病史', '脂肪肝史', '脑', '胃', '甲状腺', '肾']:
    #     c.append('is_' + str(col))
    # col = ['糖尿病史', '高血压史']
    # c.append('is_' + str(col[0]) + str(col[1]))
    #
    # for co in c:
    #     dummies = pd.get_dummies(data[[co]], columns=[co])
    #     data = pd.concat([data, dummies], axis=1)

    return data


def guke(data):
    # 骨头情况反映了年纪
    df1 = data[data['table_id'] == '3601']
    df1 = df1.groupby(['vid'], as_index=False)['field_results'].agg({'r1': np.sum})
    df1['骨质正常'] = df1['r1'].apply((lambda x: 1 if str(x).__contains__('正常') else 0))
    df1 = df1[['vid', '骨质正常']]
    data = pd.merge(data, df1, on='vid', how='left')

    # "-------------------------------one hot--------------------------------------------"
    # dummies = pd.get_dummies(data[['骨质正常']], columns=['骨质正常'])
    # data = pd.concat([data, dummies], axis=1)
    # del data['骨质正常']

    return data


def convert_tizhong(number):
    try:
        return float(number)
    except:
        return -999


def tizhong(data):
    # df1=data[data['table_id']=='2405']
    # df1['r1']=df1['field_results'].apply(convert_tizhong)
    # df1['体重范围'] = df1['r1'].apply((lambda x: 1 if x<=18.5 else 2 if (x>18.5)&(x<=23.9) else 3 if (x>23.9)&(x<=27.9) else 4 if x>27.9 else 0))
    # df1 = df1[['vid', '体重范围']]
    # data = pd.merge(data, df1, on='vid', how='left')

    # 男性女性分开来看
    df1 = data[(data['table_id'] == '2405') & (data['is_guniang'] == 1)]  # 女性
    df1['r1'] = df1['field_results'].apply(convert_tizhong)
    df1 = df1.groupby(['vid'], as_index=False)['r1'].agg({'r1': np.mean})
    df1['体重范围性别'] = df1['r1'].apply(
        lambda x: 1 if x <= 19 else 2 if (x > 19) & (x <= 24) else 3 if (x > 25) & (x <= 29) else 4 if (x >= 29) & (
        x <= 34) else 5)
    df1 = df1[['vid', '体重范围性别']]

    df2 = data[(data['table_id'] == '2405') & (data['is_guniang'] == -1)]  # 男性
    df2['r1'] = df2['field_results'].apply(convert_tizhong)
    df2 = df2.groupby(['vid'], as_index=False)['r1'].agg({'r1': np.mean})
    df2['体重范围性别'] = df2['r1'].apply((lambda x: 1 if x <= 20 else 2 if (x > 20) & (x <= 25) else 3 if (x > 26) & (
    x <= 30) else 4 if (x >= 30) & (x <= 35) else 5))
    df2 = df2[['vid', '体重范围性别']]

    df3 = pd.concat([df1, df2])
    data = pd.merge(data, df3, on='vid', how='left')

    # "-------------------------------one hot--------------------------------------------"
    # dummies = pd.get_dummies(data[['体重范围性别']], columns=['体重范围性别'])
    # data = pd.concat([data, dummies], axis=1)
    # del data['体重范围性别']

    return data


def xiongtou(data):
    df1 = data[data['table_id'] == 'A201']
    df1 = df1.groupby(['vid'], as_index=False)['field_results'].agg({'r1': np.sum})
    df1['钙化'] = df1['r1'].apply((lambda x: 1 if (str(x).__contains__('钙化')) else 0))
    df1 = df1[['vid', '钙化']]
    data = pd.merge(data, df1, on='vid', how='left')

    # "-------------------------------one hot--------------------------------------------"
    # dummies = pd.get_dummies(data[['钙化']], columns=['钙化'])
    # data = pd.concat([data, dummies], axis=1)
    # del data['钙化']

    return data


def jingzhui(data):
    df2 = data[data['table_id'] == 'A202']
    df2 = df2.groupby(['vid'], as_index=False)['field_results'].agg({'r1': np.sum})
    df2['颈椎'] = df2['r1'].apply((lambda x: 1 if (str(x).__contains__('颈椎')) | (str(x).__contains__('腰椎')) else 0))
    df2 = df2[['vid', '颈椎']]
    data = pd.merge(data, df2, on='vid', how='left')

    return data


def xinzang(data):
    df1 = data[data['table_id'] == '1001']
    df1 = df1.groupby(['vid'], as_index=False)['field_results'].agg({'r1': np.sum})
    df1['窦性心律'] = df1['r1'].apply(lambda x: 1 if (str(x).__contains__('正常')) else 2)
    df1 = df1[['vid', '窦性心律']]
    data = pd.merge(data, df1, on='vid', how='left')

    df1 = part12[part12['table_id'] == '1402']
    df1 = df1.groupby(['vid'], as_index=False)['field_results'].agg({'r1': np.sum})
    df1['血管弹性'] = df1['r1'].apply(lambda x: 1 if (str(x).__contains__('未见')) | (str(x).__contains__('正常')) else 2)
    df1 = df1[['vid', '血管弹性']]
    data = pd.merge(data, df1, on='vid', how='left')

    # "-------------------------------one hot--------------------------------------------"
    # dummies = pd.get_dummies(data[['窦性心律', '血管弹性']], columns=['窦性心律', '血管弹性'])
    # data = pd.concat([data, dummies], axis=1)
    # del data['窦性心律']
    # del data['血管弹性']

    return data


def del_onehot(data):
    del data['年龄四段']
    del data['is_guniang']

    # 彩超检查
    del data['is_肝']
    del data['is_肝1']
    del data['is_胆']
    del data['is_肾结石']
    del data['is_甲状腺彩超']
    del data['always脂肪肝']
    # 内科检查
    c = []
    c.append('is_shuhou')
    for col in ['糖尿病史', '高血压史']:
        c.append('is_' + str(col))
        c.append('is_' + str(col) + '_治疗')
    for col in ['冠心病史', '脂肪肝史', '脑', '胃', '甲状腺', '肾']:
        c.append('is_' + str(col))
    col = ['糖尿病史', '高血压史']
    c.append('is_' + str(col[0]) + str(col[1]))
    for co in c:
        del data[co]

    return data


def his_now(data):
    # data['always脂肪肝']=data.apply(lambda x:1 if ((x.is_肝==1)|(x.is_肝==2)|(x.is_肝==3)|(x.is_肝==4)|(x.is_肝==5))& (x.is_脂肪肝史==1) else 0,axis=1)

    data['always脂肪肝'] = data.apply(lambda x: 1 if (x.is_肝1 == 1) & (x.is_脂肪肝史 == 1) else 0, axis=1)
    # "-------------------------------one hot--------------------------------------------"
    # dummies = pd.get_dummies(data[['always脂肪肝']], columns=['always脂肪肝'])
    # data = pd.concat([data, dummies], axis=1)

    return data


def shuzhi(data):
    #
    data['is_number'] = data['field_results'].apply(is_number)
    datan = data[data['is_number'] == 1]
    dfx=datan.groupby(['table_id'], as_index=False)['is_number'].agg({'table_cnt': np.size})
    dfx = dfx.sort_values(by=['table_cnt'], ascending=False)
    dfx = dfx[dfx['table_cnt'] >= 20000]
    table_list = list(dfx.table_id.unique())
    #del table_list[14]

    # for col in [1815, 1814, 190, 191, 2404, 2403, 2405, 1840, 3193, 1850, 10004, 192, 1117, 193, 314, 1115, 183, 2174,
    #             10002, 10003, 37, 319, 316, 315]:
    for col in table_list:
        df1 = data[data['table_id'] == str(col)]
        df1['r' + str(col)] = df1['field_results'].apply(convert_tizhong)
        df1 = df1.groupby(['vid'], as_index=False)['r' + str(col)].agg({'r' + str(col): np.mean})
        df1 = df1[['vid', 'r' + str(col)]]
        data = pd.merge(data, df1, on='vid', how='left')
    #
    # df1 = data[data['table_id'] == '0424']
    # df1['r' + '0424'] = df1['field_results'].apply(convert_tizhong)
    # df1 = df1.groupby(['vid'], as_index=False)['r' + '0424'].agg({'r' + '0424': np.mean})
    # df1 = df1[['vid', 'r' + '0424']]
    # data = pd.merge(data, df1, on='vid', how='left')

    # df1=data[data['table_id']=='1815']
    # df1['r1815']=df1['field_results'].apply(convert_tizhong)
    # df1 = df1.groupby(['vid'], as_index=False)['r1815'].agg({'r1815': np.mean})
    # df1 = df1[['vid', 'r1815']]
    # data = pd.merge(data, df1, on='vid', how='left')
    return data


def quchong(data):
    col = [c for c in data if c not in ['field_results', 'table_id']]
    data1 = data[col]
    data1 = data1.drop_duplicates()
    return data1


def lgbCV(train, test, lie):
    col = [c for c in train if
           c not in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白', 'vid', 'table_id', 'is_number', 'vidlbl',
                     'table_idlbl']]
    X = train[col]
    y = train[lie].values
    d_train = lgb.Dataset(X, y)
    X_tes = test[col]
    y_tes = test[lie].values
    watchlist_final = lgb.Dataset(X_tes,y_tes)

    features =list(X.columns)
    cat=[]
    for f in features:
        if str(f)[0]!='r':
            cat.append(f)

    # cat=['血管弹性']

    print('Training LGBM model...')

    ### 设置初始参数--不含交叉验证参数
    print('设置参数')
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'bagging_fraction':0.8,
        'feature_fraction':0.8,
        'lambda_l1': 3.0,
        'lambda_l2': 3.0,
    }

    ### 交叉验证(调参)
    print('交叉验证')
    min_merror = float('Inf')
    best_params = {}
    "-------------------------调参--------------------------------------"
    # 准确率
    for learning_rate in [0.01]:
        print("调参1：提高准确率")
        for num_leaves in range(30,50,1):
            for max_depth in range(6, 12, 1):
                params['num_leaves'] = num_leaves
                params['max_depth'] = max_depth
                params['learning_rate'] = learning_rate

                cv_results = lgb.cv(
                    params,
                    d_train,
                    seed=2018,
                    categorical_feature=cat,
                    nfold=3,
                    metrics='rmse',
                    early_stopping_rounds=10,
                )

                mean_merror = pd.Series(cv_results['rmse-mean']).min()

                if mean_merror < min_merror:
                    min_merror = mean_merror
                    best_params['num_leaves'] = num_leaves
                    best_params['max_depth'] = max_depth
                    best_params['learning_rate'] = learning_rate

        params['num_leaves'] = best_params['num_leaves']
        params['max_depth'] = best_params['max_depth']
        params['learning_rate'] = best_params['learning_rate']

        # 过拟合
        print("调参2：降低过拟合")
        for max_bin in range(200, 255, 10):
            for min_data_in_leaf in range(10, 200, 10):
                params['max_bin'] = max_bin
                params['min_data_in_leaf'] = min_data_in_leaf
                params['learning_rate'] = learning_rate

                cv_results = lgb.cv(
                    params,
                    d_train,
                    seed=2018,
                    categorical_feature=cat,
                    nfold=3,
                    metrics='rmse',
                    early_stopping_rounds=10,
                )

                mean_merror = pd.Series(cv_results['rmse-mean']).min()

                if mean_merror < min_merror:
                    min_merror = mean_merror
                    best_params['max_bin'] = max_bin
                    best_params['min_data_in_leaf'] = min_data_in_leaf
                    best_params['learning_rate'] = learning_rate

        params['min_data_in_leaf'] = best_params['min_data_in_leaf']
        params['max_bin'] = best_params['max_bin']
        params['learning_rate'] = best_params['learning_rate']

        print("调参3：降低过拟合")
        for feature_fraction in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            for bagging_fraction in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                params['feature_fraction'] = feature_fraction
                params['bagging_fraction'] = bagging_fraction
                params['learning_rate'] = learning_rate

                cv_results = lgb.cv(
                    params,
                    d_train,
                    seed=2018,
                    categorical_feature=cat,
                    nfold=3,
                    metrics='rmse',
                    early_stopping_rounds=10,
                )

                mean_merror = pd.Series(cv_results['rmse-mean']).min()

                if mean_merror < min_merror:
                    min_merror = mean_merror
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
                    best_params['learning_rate'] = learning_rate

        params['feature_fraction'] = best_params['feature_fraction']
        params['bagging_fraction'] = best_params['bagging_fraction']
        params['learning_rate'] = best_params['learning_rate']

        print("调参4：降低过拟合")
        for lambda_l1 in [0.5,1,1.5,2,2.5,3,3.5,4]:
            for lambda_l2 in [0.5,1,1.5,2,2.5,3,3.5,4]:
                params['lambda_l1'] = lambda_l1
                params['lambda_l2'] = lambda_l2
                params['learning_rate'] = learning_rate

                cv_results = lgb.cv(
                    params,
                    d_train,
                    categorical_feature=cat,
                    seed=2018,
                    nfold=3,
                    metrics='rmse',
                    early_stopping_rounds=10,
                )

                mean_merror = pd.Series(cv_results['rmse-mean']).min()
                if mean_merror < min_merror:
                    min_merror = mean_merror
                    best_params['lambda_l1'] = lambda_l1
                    best_params['lambda_l2'] = lambda_l2
                    best_params['learning_rate'] = learning_rate

        params['lambda_l1'] = best_params['lambda_l1']
        params['lambda_l2'] = best_params['lambda_l2']
        params['learning_rate'] = best_params['learning_rate']

    ### 训练
    # params['learning_rate'] = learning_rate

    print(params)
    # using lgb_test as valid_sets
    lgb_model= lgb.train(params,train_set=d_train, num_boost_round=2000, valid_sets=watchlist_final, categorical_feature=cat,verbose_eval=5,early_stopping_rounds=200)


    #lgb_model= lgb.train(params, train_set=d_train, num_boost_round=2000, valid_sets=watchlist_final, verbose_eval=5,early_stopping_rounds=200)
    best_iter=lgb_model.best_iteration

    predictors = [i for i in X.columns]
    feat_imp = pd.Series(lgb_model.feature_importance(importance_type='gain', iteration=-1), predictors).sort_values(ascending=False)
    print(feat_imp)
    print('将特征重要性输出到excel里')
    feat_imp.to_csv('特征重要性/特征重要性0418.csv')
    print(feat_imp.shape)
    # pred= lgb_model.predict(test[col])
    pred = lgb_model.predict(test[col])
    test['pred' + str(lie)] = pred  # 保留三位小数
    test['pred' + str(lie)] = test['pred' + str(lie)].apply(lambda x: round(x, 3))

    print('按照vid进行groupby')
    tmp = test.groupby(['vid'], as_index=False)[lie].agg({str(lie): np.mean})
    tmp1 = test.groupby(['vid'], as_index=False)['pred' + str(lie)].agg({'pred' + str(lie): np.mean})
    tmp = pd.merge(tmp, tmp1, on='vid', how='left')
   # print(tmp)
    print('误差值')
    tmp['wucha'] = (np.log(tmp[lie] + 1) - np.log(tmp['pred' + str(lie)] + 1)) ** 2
    wucha = sum(tmp['wucha'] / len(tmp))
    print(wucha)
    return test, best_iter, wucha


def sub(train, test, lie, iters):
    col = [c for c in train if
           c not in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白', 'vid', 'table_id', 'is_number', 'vidlbl']]
    X = train[col]
    y = train[lie].values

    d_train = lgb.Dataset(X, y)
    watchlist_final = lgb.Dataset(X, y)
    print('Training LGBM model...')

    features =list(X.columns)
    cat=[]
    for f in features:
        if str(f)[0]!='r':
            cat.append(f)

    ### 设置初始参数--不含交叉验证参数
    print('设置参数')
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'lambda_l1': 3.0,
        'lambda_l2': 3.0,
    }

    ### 交叉验证(调参)
    print('交叉验证')
    min_merror = float('Inf')
    #best_params = {'min_data_in_leaf':100}
    best_params = {'min_data_in_leaf':50,
                   'max_bin':10,
                   'learning_rate':0.01,
                   'feature_fraction':0.9,
                   'bagging_fraction':0.9
                   }
    "-------------------------调参--------------------------------------"
    # 准确率
    for learning_rate in [0.01]:
        print("调参1：提高准确率")
        for num_leaves in range(30, 50, 1):
            for max_depth in range(6, 12, 1):
                params['num_leaves'] = num_leaves
                params['max_depth'] = max_depth
                params['learning_rate'] = learning_rate

                cv_results = lgb.cv(
                    params,
                    d_train,
                    seed=2018,
                    nfold=3,
                    categorical_feature=cat,
                    metrics='rmse',
                    early_stopping_rounds=10,
                )

                mean_merror = pd.Series(cv_results['rmse-mean']).min()

                if mean_merror < min_merror:
                    min_merror = mean_merror
                    best_params['num_leaves'] = num_leaves
                    best_params['max_depth'] = max_depth
                    best_params['learning_rate'] = learning_rate

        params['num_leaves'] = best_params['num_leaves']
        params['max_depth'] = best_params['max_depth']
        params['learning_rate'] = best_params['learning_rate']

        # 过拟合
        print("调参2：降低过拟合")
        for max_bin in range(200, 255, 10):
            for min_data_in_leaf in range(10, 200, 10):
                params['max_bin'] = max_bin
                params['min_data_in_leaf'] = min_data_in_leaf
                params['learning_rate'] = learning_rate

                cv_results = lgb.cv(
                    params,
                    d_train,
                    seed=2018,
                    categorical_feature=cat,
                    nfold=3,
                    metrics='rmse',
                    early_stopping_rounds=10,
                )

                mean_merror = pd.Series(cv_results['rmse-mean']).min()

                if mean_merror < min_merror:
                    min_merror = mean_merror
                    best_params['max_bin'] = max_bin
                    best_params['min_data_in_leaf'] = min_data_in_leaf
                    best_params['learning_rate'] = learning_rate

        params['min_data_in_leaf'] = best_params['min_data_in_leaf']
        params['max_bin'] = best_params['max_bin']
        params['learning_rate'] = best_params['learning_rate']

        print("调参3：降低过拟合")
        for feature_fraction in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            for bagging_fraction in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                params['feature_fraction'] = feature_fraction
                params['bagging_fraction'] = bagging_fraction
                params['learning_rate'] = learning_rate

                cv_results = lgb.cv(
                    params,
                    d_train,
                    seed=2018,
                    categorical_feature=cat,
                    nfold=3,
                    metrics='rmse',
                    early_stopping_rounds=10,
                )

                mean_merror = pd.Series(cv_results['rmse-mean']).min()

                if mean_merror < min_merror:
                    min_merror = mean_merror
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
                    best_params['learning_rate'] = learning_rate

        params['feature_fraction'] = best_params['feature_fraction']
        params['bagging_fraction'] = best_params['bagging_fraction']
        params['learning_rate'] = best_params['learning_rate']

        print("调参4：降低过拟合")
        for lambda_l1 in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
            for lambda_l2 in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
                params['lambda_l1'] = lambda_l1
                params['lambda_l2'] = lambda_l2
                params['learning_rate'] = learning_rate

                cv_results = lgb.cv(
                    params,
                    d_train,
                    seed=2018,
                    categorical_feature=cat,
                    nfold=3,
                    metrics='rmse',
                    early_stopping_rounds=10,
                )

                mean_merror = pd.Series(cv_results['rmse-mean']).min()
                if mean_merror < min_merror:
                    min_merror = mean_merror
                    best_params['lambda_l1'] = lambda_l1
                    best_params['lambda_l2'] = lambda_l2
                    best_params['learning_rate'] = learning_rate

        params['lambda_l1'] = best_params['lambda_l1']
        params['lambda_l2'] = best_params['lambda_l2']
        params['learning_rate'] = best_params['learning_rate']

    ### 训练
    # params['learning_rate'] = learning_rate

    print(params)
    # using lgb_test as valid_sets
    lgb_model = lgb.train(params, train_set=d_train, num_boost_round=iters, categorical_feature=cat,valid_sets=watchlist_final, verbose_eval=5)

    pred = lgb_model.predict(test[col])
    test[lie] = pred
    test[lie] = test[lie].apply(lambda x: round(x, 3))
    return test


if __name__ == "__main__":
    print('读入part12')
    part_1 = pd.read_csv('input/meinian_round1_data_part1_20180408.txt', sep='$')
    part_2 = pd.read_csv('input/meinian_round1_data_part2_20180408.txt', sep='$')
    print('part12拼接')
    part12 = pd.concat([part_1, part_2])
    print('part12的shape', part12.shape)
    print('去掉result为空的')
    part12 = part12[part12.field_results.notnull()]
    print('part12的shape', part12.shape)

    print('做特征，是否为脂肪肝')
    # part12 = zhifanggan(part12)
    # part12=cnt_feature(part12)
    "----------------------------------------"
    part12 = caichao(part12)
    part12 = jiankang(part12)
    part12 = sex(part12)
    part12 = neike(part12)
    part12 = guke(part12)
    part12 = tizhong(part12)
    part12 = xiongtou(part12)

    part12 = xinzang(part12)

    part12 = his_now(part12)

    #part12 = del_onehot(part12)
    part12 = shuzhi(part12)
    "--------------------------------------"

    # 反向增加特征
    # part12=xinzang(part12)
    # part12=jingzhui(part12)

    print('只包括数字的part12')
    # part12['is_number'] = part12['field_results'].apply(is_number)
    # part12n = part12[part12['is_number'] == 1]
    # print('part12n的长度',part12n.shape)

    print("对part12去重")
    part12n = quchong(part12)

    print('读入train和test')
    train = pd.read_csv('input/meinian_round1_train_20180408.csv')
    print('train的最原始长度', train.shape)
    for col in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']:
        train[col] = train[col].apply(conver_label)
        train = train[train[col] != -999]

    test = pd.read_csv('input/meinian_round1_test_b_20180505.csv',encoding='gbk')
    print('train去掉label为空的的长度', train.shape)
    print('test的长度', test.shape)

    a = len(train)
    b = len(test)  # 为了线下划分验证集

    print('part12n与train,test合并')
    train = pd.merge(train, part12n, on='vid', how='left')
    test = pd.merge(test, part12n, on='vid', how='left')
    train, test = pre_process(train, test)
    print('合并之后train的长度', train.shape)
    print('合并之后test的长度', test.shape)

    print('去掉train异常值')
    train = train[(train['收缩压'] > 90) & (train['收缩压'] < 180)]#0.008
    #train = train[(train['收缩压'] > 140) & (train['收缩压'] < 200)]
    #train = train[(train['收缩压'] > 90) & (train['收缩压'] < 200)] #0.0161
    #train = train[(train['收缩压'] > 80) & (train['收缩压'] < 200)]
    train = train[(train['舒张压'] < 110) & (train['舒张压'] > 55)]
    train = train[(train['血清甘油三酯'] > 0) & (train['血清甘油三酯'] < 10)]
    train = train[(train['血清高密度脂蛋白'] > 0.75) & (train['血清高密度脂蛋白'] < 2.5)]
    train = train[(train['血清低密度脂蛋白'] > 0.9) & (train['血清低密度脂蛋白'] < 5.5)]
    "----------------------------------------------------线下----------------------------------------"
    '按照vid进行划分'
    vidlist = list(train.vid.unique())
    trainvid = vidlist[:a - b]
    testvid = vidlist[a - b:]
    print('线下train和test vid 的长度')
    print(len(trainvid))
    print(len(testvid))
    print('按照trainvid和testvid来划分线下验证集')
    trainoff = train[train.vid.isin(trainvid)]
    testoff = train[train.vid.isin(testvid)]
    print('划分之后train和test的长度')
    print(trainoff.shape)
    print(testoff.shape)

    iterss = []  # 储存五次的最佳迭代次数
    wuchas = []
    for lie in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']:
    #for lie in ['收缩压']:
        testoff, iters, wucha = lgbCV(trainoff, testoff, lie)
        iterss.append(iters)
        wuchas.append(wucha)
    print('线下五次的误差')
    print(wuchas)
    print('五次平均误差')
    print(sum(wuchas) / len(wuchas))

    "----------------------------------------------------线上----------------------------------------"
    i=0
    for lie in ['收缩压','舒张压', '血清甘油三酯', '血清高密度脂蛋白','血清低密度脂蛋白']:
        iters=iterss[i]
        test=sub(train, test,lie,iters)
        i=i+1

    print('test没处理的长度',test.shape)
    #test.to_csv('result/r0412a.csv', index=False)
    print('按照vid取平均')
    for lie in ['收缩压']:
        tmp=test.groupby(['vid'],as_index=False)[lie].agg({str(lie): np.mean})
    for lie in ['舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']:
        tmp1= test.groupby(['vid'], as_index=False)[lie].agg({str(lie): np.mean})
        tmp=pd.merge(tmp1,tmp,on='vid',how='left')

    print('test处理的长度', tmp.shape)
    #tmp.to_csv('result/r0413.csv', index=False)
    "---------------------------------------------------结果提交---------------------------------"
    sub = pd.read_csv('input/meinian_round1_test_b_20180505.csv',encoding='gbk')
    for col in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']:
        del sub[col]

    for col in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']:
        sub = pd.merge(sub, tmp[['vid', col]], on='vid', how='left')

    print('转化为小数点后三位')
    for col in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']:
        sub[col] = sub[col].apply(lambda x: round(x, 3))

    sub.to_csv('result/sub0506a.csv', index=False, header=None)
    # #


