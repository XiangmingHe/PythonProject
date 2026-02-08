# 导包
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier  # 随机森林  # AdaBoost分类模型API
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from xgboost import XGBClassifier,XGBRegressor    # XGBoost分类模型API
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import os
import logging
import matplotlib.pyplot as plt

# =================设置中文显示=================
# plt.rcParams['font.family'] = 'SimHei'
# plt.rcParams['font.size'] = 10
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1.数据准备
# 1.1读取数据
def load_data(df_train,df_test):
    """
    加载并预处理数据
    :param df_train: 训练集 DataFrame
    :param df_test: 测试集 DataFrame
    :return: 处理后的训练特征、标签和测试特征、标签
    """
    print(df_train.shape)
    print(df_train.isna().sum())
    print('----------加载数据----------------')
    # 1.2提取特征和标签
    X_train = df_train.iloc[:,1:]
    Y_train = df_train.iloc[:,0]
    X_test = df_test.iloc[:,0:-1]
    Y_test = df_test['Attrition']
    X_train = X_train.drop(columns=['EmployeeNumber','Over18', 'StandardHours'])
    X_test = X_test.drop(columns=['EmployeeNumber','Over18', 'StandardHours'])
    print(X_train.shape)
    print(X_test.shape)
    logging.info("数据加载完成")
    return X_train,X_test,Y_train,Y_test

# label encoder和one encoder分别对类别数据进行特征编码，处理组合后的数据特征后形成特征向量
def apply_label_encoding(X_train, X_test):
    """
    对类别特征进行 Label Encoding
    :param X_train: 训练集特征 DataFrame
    :param X_test: 测试集特征 DataFrame
    :return: 编码后的训练集和测试集特征
    """
    le = LabelEncoder()
    categorical_columns = X_train.select_dtypes(include=['object']).columns  # 获取类别列

    for col in categorical_columns:
        # 合并训练集和测试集以确保一致性
        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined.astype(str))  # 拟合所有唯一值

        # 转换训练集和测试集
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    return X_train, X_test

# 1.3 对特征进行独热编码，并确保训练集和测试集列对齐
# def preprocess_data(X_train,X_test):
#     """
#     数据预处理,对特征进行独热编码，并确保训练集和测试集列对齐
#     :param X_train: 训练集特征
#     :param X_test: 测试集特征
#     :param Y_train: 训练集标签
#     :param Y_test: 测试集标签
#     :return: 处理后的训练特征、标签和测试特征、标签
#     """
#     # 1.3 对特征进行独热编码，并确保训练集和测试集列对齐
#     x_train = pd.get_dummies(X_train,drop_first=True)
#     x_test = pd.get_dummies(X_test,drop_first=True)
#     print(x_train.shape,x_train)
#     print('----------特征和标签处理----------------')
#     return x_train,x_test

# 1.5 特征标准化
def standardize_features(x_train,x_test):
    """
    对特征进行标准化
    :param x_train: 训练集特征
    :param x_test: 测试集特征
    :return: 标准化后的训练集和测试集特征
    """
    ss = StandardScaler()
    x_train_ss = ss.fit_transform(x_train)
    x_test_ss = ss.transform(x_test)
    return x_train_ss,x_test_ss ,ss


# # 2.模型评估
def train_and_evaluate(model, x_train_ss, x_test_ss, Y_train, Y_test, model_name="Model"):
    """
    通用模型训练与评估函数
    :param model: 模型对象
    :param x_train_ss: 标准化后的训练集特征
    :param x_test_ss: 标准化后的测试集特征
    :param Y_train: 训练集标签
    :param Y_test: 测试集标签
    :param model_name: 模型名称（用于日志输出）
    :return: 预测结果和概率
    """
    model.fit(x_train_ss, Y_train)
    y_pred = model.predict(x_test_ss)
    y_pred_proba = model.predict_proba(x_test_ss)[:, 1]

    auc_score = roc_auc_score(Y_test, y_pred_proba)
    f1 = f1_score(Y_test, y_pred)
    print(f"{model_name} AUC分数: {auc_score}, F1分数: {f1}")
    return y_pred, y_pred_proba

def plot_feature_importance(x_train_ss, Y_train): # 特征重要性可视化
    """
    特征重要性可视化
    :param x_train_ss: 标准化后的训练集特征
    :param Y_train: 训练集标签
    :param feature_names: 特征名列表
    """
    mutual_info = mutual_info_classif(x_train_ss,Y_train,random_state=666)
    mutual_info_series = pd.Series(mutual_info,index=x_train.columns)
    mutual_info_sorted = mutual_info_series.sort_values(ascending=True)

    plt.figure(figsize=(12, 9))
    plt.title('特征重要性',fontsize=10)
    mutual_info_sorted.plot(kind='bar', color='r')
    plt.xlabel('特征名称')
    plt.ylabel('特互信息得分')
    plt.xticks(rotation=45, ha='right') # 旋转x轴标签以便阅读
    plt.tight_layout()
    plt.show()

# ------------------- 网格搜索优化逻辑回归模型 -------------------
def optimize_logistic_regression_with_grid_search(x_train_ss, x_test_ss, Y_train, Y_test):
    """
    使用网格搜索优化逻辑回归模型
    :param x_train: 训练集特征
    :param x_test: 测试集特征
    :param Y_train: 训练集标签
    :param Y_test: 测试集标签
    :return: 最优模型和预测结果
    """
    # 定义逻辑回归模型
    lr_model = LogisticRegression(random_state=666, max_iter=1000)

    # 定义参数网格
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # 正则化强度
        'penalty': ['l1', 'l2'],       # 正则化类型
        'solver': ['liblinear']}        # 仅支持 l1 和 l2 的求解器

    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(estimator=lr_model,param_grid=param_grid,scoring='roc_auc',cv=5,verbose=1,n_jobs=-1)

    # 拟合模型
    grid_search.fit(x_train_ss, Y_train)

    # 输出最优参数
    print("最优参数:", grid_search.best_params_)
    # print("最优得分:", grid_search.best_score_)

    # 使用最优模型进行预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test_ss)
    y_pred_proba = best_model.predict_proba(x_test_ss)[:, 1]

    # 评估模型性能
    auc_score = roc_auc_score(Y_test, y_pred_proba)
    f1 = f1_score(Y_test, y_pred)
    print(f"优化后逻辑回归 AUC分数: {auc_score}, F1分数: {f1}")

    return best_model, y_pred, y_pred_proba

# ------------------- 网格搜索优化XGBoost模型 -------------------
def optimize_xgboost_with_grid_search(x_train_ss, x_test_ss, Y_train, Y_test):
    """
    使用网格搜索优化XGBoost模型
    :param x_train: 训练集特征
    :param x_test: 测试集特征
    :param Y_train: 训练集标签
    :param Y_test: 测试集标签
    :return: 最优模型和预测结果
    """
    # 定义XGBoost模型
    xgb_model = XGBClassifier(random_state=666, eval_metric='logloss')

    # 定义参数网格
    param_grid = {
        'n_estimators': [50,55,60,65,75,90,100,110,120,125,130,135,140,150],'learning_rate': [0.01, 0.1, 0.2],'max_depth': [1,3, 5, 7,9],
        'subsample': [0.8, 1.0]}

    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(estimator=xgb_model,param_grid=param_grid,scoring='roc_auc',cv=5,verbose=1,n_jobs=-1)

    # 拟合模型
    grid_search.fit(x_train_ss, Y_train)

    # 输出最优参数
    print("最优参数:", grid_search.best_params_)
    # print("最优得分:", grid_search.best_score_)

    # 使用最优模型进行预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test_ss)
    y_pred_proba = best_model.predict_proba(x_test_ss)[:, 1]

    # 评估模型性能
    auc_score = roc_auc_score(Y_test, y_pred_proba)
    f1 = f1_score(Y_test, y_pred)
    print(f"优化后XGBoost AUC分数: {auc_score}, F1分数: {f1}")

    return best_model, y_pred, y_pred_proba

# ------------------- 网格搜索优化随机森林模型 -------------------
def optimize_random_forest_with_grid_search(x_train, x_test, Y_train, Y_test):
    """
    使用网格搜索优化随机森林模型
    :param x_train: 训练集特征
    :param x_test: 测试集特征
    :param Y_train: 训练集标签
    :param Y_test: 测试集标签
    :return: 最优模型和预测结果
    """
    # 定义随机森林模型
    rf_model = RandomForestClassifier(random_state=666)

    # 定义参数网格
    param_grid = {
        'n_estimators': [100, 200, 300],'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4]}

    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(estimator=rf_model,param_grid=param_grid,
        scoring='roc_auc',cv=5,verbose=1,n_jobs=-1)

    # 拟合模型
    grid_search.fit(x_train_ss, Y_train)

    # 输出最优参数
    print("最优参数:", grid_search.best_params_)
    # print("最优得分:", grid_search.best_score_)

    # 使用最优模型进行预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test_ss)
    y_pred_proba = best_model.predict_proba(x_test_ss)[:, 1]

    # 评估模型性能
    auc_score = roc_auc_score(Y_test, y_pred_proba)
    f1 = f1_score(Y_test, y_pred)
    print(f"优化后随机森林 AUC分数: {auc_score}, F1分数: {f1}")

    return best_model, y_pred, y_pred_proba

# ------------------- 网格搜索优化AdaBoost模型 -------------------
def optimize_adaboost_with_grid_search(x_train, x_test, Y_train, Y_test):
    """
    使用网格搜索优化AdaBoost模型
    :param x_train: 训练集特征
    :param x_test: 测试集特征
    :param Y_train: 训练集标签
    :param Y_test: 测试集标签
    :return: 最优模型和预测结果
    """
    # 定义AdaBoost模型
    ada_model = AdaBoostClassifier(random_state=666)

    # 定义参数网格
    param_grid = {'n_estimators': [50,75, 100.125,150],'learning_rate': [0.01, 0.1, 1.0]}

    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(estimator=ada_model,param_grid=param_grid,
        scoring='roc_auc',cv=5,verbose=1,n_jobs=-1)

    # 拟合模型
    grid_search.fit(x_train_ss, Y_train)

    # 输出最优参数
    print("最优参数:", grid_search.best_params_)
    # print("最优得分:", grid_search.best_score_)

    # 使用最优模型进行预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test_ss)
    y_pred_proba = best_model.predict_proba(x_test_ss)[:, 1]

    # 评估模型性能
    auc_score = roc_auc_score(Y_test, y_pred_proba)
    f1 = f1_score(Y_test, y_pred)
    print(f"优化后AdaBoost AUC分数: {auc_score}, F1分数: {f1}")

    return best_model, y_pred, y_pred_proba




# ------------------- 主程序入口 -------------------
# ============ 新增：ROC曲线合并绘制函数 ============

def plot_merged_roc_curves(y_true, predictions_dict, model_names_dict=None,
                           save_path='../data/picture/人才流失_多模型ROC合并曲线.png',
                           figsize=(14, 10), show_best_threshold=True):
    """
    合并绘制多个模型的ROC曲线对比图

    参数:
    ----------
    y_true : array-like
        真实标签
    predictions_dict : dict
        预测概率字典，格式: {'模型名': y_pred_proba}
    model_names_dict : dict, optional
        模型显示名称字典，用于美化显示
    save_path : str
        保存路径
    figsize : tuple
        图形尺寸
    show_best_threshold : bool
        是否显示最佳阈值点

    返回:
    -------
    dict
        各模型的详细评估结果
    """
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    plt.rcParams['font.sans-serif'] = [
        'Arial Unicode MS',  # macOS 自带中文字体
        'PingFang SC',  # 苹方字体
        'Hiragino Sans GB',  # 冬青黑体
        'STHeiti',  # 华文黑体
        'Lantinghei SC'  # 兰亭黑
    ]
    plt.rcParams['axes.unicode_minus'] = False
    print("\n" + "=" * 70)
    print("绘制多模型ROC合并曲线")
    print("=" * 70)

    # 创建图形
    plt.figure(figsize=figsize)

    # 颜色和样式配置
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FFA07A', '#20B2AA']
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']

    # 存储结果
    results = {}
    auc_scores = []

    # 绘制每个模型的ROC曲线
    for idx, (model_key, y_pred_proba) in enumerate(predictions_dict.items()):
        try:
            # 使用美化后的模型名称
            if model_names_dict and model_key in model_names_dict:
                model_name = model_names_dict[model_key]
            else:
                model_name = model_key

            # 计算ROC曲线
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            auc_score = roc_auc_score(y_true, y_pred_proba)

            # 计算最佳阈值（Youden's J指数）
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
            best_threshold = thresholds[best_idx]

            # 选择颜色和样式
            color = colors[idx % len(colors)]
            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]

            # 绘制ROC曲线
            plt.plot(fpr, tpr,
                     color=color,
                     linestyle=line_style,
                     linewidth=2.5,
                     alpha=0.85,
                     label=f'{model_name} (AUC={auc_score:.3f})')

            # 标记最佳阈值点
            if show_best_threshold:
                plt.scatter(fpr[best_idx], tpr[best_idx],
                            color=color,
                            s=100,
                            marker=marker,
                            edgecolors='black',
                            linewidth=1.5,
                            zorder=5,
                            alpha=0.9,
                            label=f'{model_name}最佳阈值点' if idx == 0 else "")

            # 保存结果
            results[model_name] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'roc_auc': roc_auc,
                'auc_score': auc_score,
                'best_threshold': best_threshold,
                'best_tpr': tpr[best_idx],
                'best_fpr': fpr[best_idx],
                'color': color,
                'line_style': line_style
            }

            auc_scores.append(auc_score)
            print(f"  ✅ {model_name}: AUC = {auc_score:.4f}, 最佳阈值 = {best_threshold:.3f}")

        except Exception as e:
            print(f"  ❌ {model_key} 绘制失败: {str(e)[:50]}")
            continue

    # 绘制随机猜测线
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6, label='随机猜测 (AUC=0.5000)')

    # 设置图形属性
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('假正例率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正例率 (True Positive Rate)', fontsize=12)
    plt.title('人才流失预测模型ROC曲线对比', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')

    # 添加性能区域阴影
    plt.fill_between([0, 1], [0, 0.7], [0.7, 0.7], alpha=0.05, color='red', label='差 (AUC<0.7)')
    plt.fill_between([0, 1], [0.7, 0.8], [0.8, 0.8], alpha=0.05, color='orange', label='一般 (0.7≤AUC<0.8)')
    plt.fill_between([0, 1], [0.8, 0.9], [0.9, 0.9], alpha=0.05, color='yellow', label='良好 (0.8≤AUC<0.9)')
    plt.fill_between([0, 1], [0.9, 1.0], [1.0, 1.0], alpha=0.05, color='green', label='优秀 (AUC≥0.9)')

    # 添加统计信息
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['auc_score'])[0]
        best_auc = results[best_model]['auc_score']
        avg_auc = np.mean(auc_scores)

        stats_text = (f'📊 模型性能统计\n'
                      f'• 模型数量: {len(results)}\n'
                      f'• 最佳模型: {best_model}\n'
                      f'• 最佳AUC: {best_auc:.4f}\n'
                      f'• 平均AUC: {avg_auc:.4f}\n'
                      f'• 样本数量: {len(y_true)}\n'
                      f'• 离职率: {y_true.mean():.2%}')

        plt.text(0.95, 0.05, stats_text,
                 transform=plt.gca().transAxes,
                 fontsize=10,
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # 保存图形
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"\n📈 分析完成!")
    print(f"✅ 合并图表已保存到: {save_path}")
    print("=" * 70)

    return results


# ============ 新增：保存详细评估报告函数 ============

def save_model_evaluation_report(results, y_test, save_path='../data/picture/人才流失模型评估报告.csv'):
    """
    保存模型评估详细报告
    """
    import pandas as pd

    report_data = []

    for model_name, result in results.items():
        report_data.append({
            '模型名称': model_name,
            'AUC分数': result['auc_score'],
            '最佳阈值': result['best_threshold'],
            '真正例率(TPR)': result['best_tpr'],
            '假正例率(FPR)': result['best_fpr'],
            '特异度': 1 - result['best_fpr'],
            'Youden J指数': result['best_tpr'] - result['best_fpr']
        })

    report_df = pd.DataFrame(report_data)
    report_df = report_df.sort_values('AUC分数', ascending=False)

    # 添加性能评级
    def get_performance_rating(auc_score):
        if auc_score >= 0.9:
            return '★★★★★ (优秀)'
        elif auc_score >= 0.8:
            return '★★★★ (良好)'
        elif auc_score >= 0.7:
            return '★★★ (一般)'
        else:
            return '★★ (需改进)'

    report_df['性能评级'] = report_df['AUC分数'].apply(get_performance_rating)

    # 保存报告
    report_df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print("📋 详细评估报告:")
    print(report_df.to_string(index=False))
    print(f"\n✅ 详细报告已保存到: {save_path}")

    return report_df


# ============ 修改主程序：添加ROC合并曲线 ============

# ------------------- 主程序入口 -------------------
if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv', sep=',')
    df_test = pd.read_csv('../data/test2.csv', sep=',')

    # 数据准备
    X_train, X_test, Y_train, Y_test = load_data(df_train, df_test)
    x_train, x_test = apply_label_encoding(X_train, X_test)
    x_train_ss, x_test_ss, ss = standardize_features(x_train, x_test)

    # ============ 基础模型训练 ============
    print("\n" + "=" * 70)
    print("基础模型训练")
    print("=" * 70)

    # 模型训练：逻辑回归
    lr_model = LogisticRegression(random_state=666)
    y_pred_lr, y_pred_proba_lr = train_and_evaluate(lr_model, x_train_ss, x_test_ss, Y_train, Y_test, "逻辑回归")

    # 模型训练：XGBoost
    xgb_model = XGBClassifier(n_estimators=100, random_state=666, learning_rate=0.1, use_label_encoder=False)
    y_pred_xgb, y_pred_proba_xgb = train_and_evaluate(xgb_model, x_train_ss, x_test_ss, Y_train, Y_test, "XGBoost")

    # 模型训练：随机森林
    rf_model = RandomForestClassifier(n_estimators=100, random_state=666, max_depth=None)
    y_pred_rf, y_pred_proba_rf = train_and_evaluate(rf_model, x_train_ss, x_test_ss, Y_train, Y_test, "随机森林")

    # 模型训练：决策树
    dt_model = DecisionTreeClassifier(random_state=666)
    y_pred_dt, y_pred_proba_dt = train_and_evaluate(dt_model, x_train_ss, x_test_ss, Y_train, Y_test, "决策树")

    # 模型训练：AdaBoost
    ada_model = AdaBoostClassifier(n_estimators=50, random_state=666, learning_rate=0.1)
    y_pred_ada, y_pred_proba_ada = train_and_evaluate(ada_model, x_train_ss, x_test_ss, Y_train, Y_test, "AdaBoost")

    # ============ 优化模型训练 ============
    print("\n" + "=" * 70)
    print("优化模型训练")
    print("=" * 70)

    # 网格搜索优化逻辑回归模型
    best_lr_model, y_pred_lr_optimized, y_pred_proba_lr_optimized = optimize_logistic_regression_with_grid_search(
        x_train_ss, x_test_ss, Y_train, Y_test)

    # 网格搜索优化XGBoost模型
    best_xgb_model, y_pred_xgb_optimized, y_pred_proba_xgb_optimized = optimize_xgboost_with_grid_search(
        x_train_ss, x_test_ss, Y_train, Y_test)

    # 网格搜索优化随机森林模型
    best_rf_model, y_pred_rf_optimized, y_pred_proba_rf_optimized = optimize_random_forest_with_grid_search(
        x_train_ss, x_test_ss, Y_train, Y_test)

    # 网格搜索优化AdaBoost模型
    best_ada_model, y_pred_ada_optimized, y_pred_proba_ada_optimized = optimize_adaboost_with_grid_search(
        x_train_ss, x_test_ss, Y_train, Y_test)

    # ============ 特征重要性可视化 ============
    print("\n" + "=" * 70)
    print("特征重要性分析")
    print("=" * 70)
    plot_feature_importance(x_train_ss, Y_train)

    # ============ 绘制ROC合并曲线 ============
    print("\n" + "=" * 70)
    print("多模型ROC曲线合并分析")
    print("=" * 70)

    # 准备所有模型的预测概率
    predictions_dict = {
        '逻辑回归': y_pred_proba_lr,
        'XGBoost': y_pred_proba_xgb,
        '随机森林': y_pred_proba_rf,
        '决策树': y_pred_proba_dt,
        'AdaBoost': y_pred_proba_ada,
        '优化逻辑回归': y_pred_proba_lr_optimized,
        '优化XGBoost': y_pred_proba_xgb_optimized,
        '优化随机森林': y_pred_proba_rf_optimized,
        '优化AdaBoost': y_pred_proba_ada_optimized
    }

    # 美化模型名称显示
    model_names_dict = {
        '逻辑回归': '逻辑回归 (基础)',
        'XGBoost': 'XGBoost (基础)',
        '随机森林': '随机森林 (基础)',
        '决策树': '决策树 (基础)',
        'AdaBoost': 'AdaBoost (基础)',
        '优化逻辑回归': '逻辑回归 (优化)',
        '优化XGBoost': 'XGBoost (优化)',
        '优化随机森林': '随机森林 (优化)',
        '优化AdaBoost': 'AdaBoost (优化)'
    }

    # 绘制合并ROC曲线
    roc_results = plot_merged_roc_curves(
        y_true=Y_test,
        predictions_dict=predictions_dict,
        model_names_dict=model_names_dict,
        save_path='../data/picture/人才流失_多模型ROC合并曲线.png',
        figsize=(16, 12),
        show_best_threshold=True
    )

    # ============ 保存详细评估报告 ============
    print("\n" + "=" * 70)
    print("生成详细评估报告")
    print("=" * 70)

    if roc_results:
        report_df = save_model_evaluation_report(
            roc_results,
            Y_test,
            save_path='../data/picture/人才流失模型评估报告.csv'
        )

        # 输出最佳模型推荐
        best_model_row = report_df.iloc[0]
        print("\n" + "=" * 70)
        print("🏆 最佳模型推荐")
        print("=" * 70)
        print(f"模型名称: {best_model_row['模型名称']}")
        print(f"AUC分数: {best_model_row['AUC分数']:.4f}")
        print(f"性能评级: {best_model_row['性能评级']}")
        print(f"建议阈值: {best_model_row['最佳阈值']:.3f}")
        print(f"预测性能:")
        print(f"  • 真正例率(TPR): {best_model_row['真正例率(TPR)']:.3f}")
        print(f"  • 假正例率(FPR): {best_model_row['假正例率(FPR)']:.3f}")
        print(f"  • 特异度: {best_model_row['特异度']:.3f}")
        print("=" * 70)

    print("\n🎉 人才流失预测模型分析完成!")
























    # 网格搜索优化XGBoost模型
    best_xgb_model, y_pred_xgb_optimized, y_pred_proba_xgb_optimized = optimize_xgboost_with_grid_search(
    x_train_ss, x_test_ss, Y_train, Y_test)

    # 网格搜索优化随机森林模型
    best_rf_model, y_pred_rf_optimized, y_pred_proba_rf_optimized = optimize_random_forest_with_grid_search(
    x_train_ss, x_test_ss, Y_train, Y_test)

    # 网格搜索优化AdaBoost模型
    best_ada_model, y_pred_ada_optimized, y_pred_proba_ada_optimized = optimize_adaboost_with_grid_search(
    x_train_ss, x_test_ss, Y_train, Y_test)
