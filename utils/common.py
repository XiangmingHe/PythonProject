# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from openpyxl.utils import datetime


def data_preprocessing(path):
    """
    1.获取数据源
    2.时间格式化，转为2024-12-20 09:00:00这种格式
    3.按时间升序排序
    4.去重
    :param path:
    :return:
    """

    # 定义中英文对照字典
    column_mapping = {
        'Attrition': '是否离职',
        'Age': '年龄',
        'BusinessTravel': '出差频率',
        'Department': '部门',
        'DistanceFromHome': '通勤距离',
        'Education': '教育程度',
        'EducationField': '专业领域',
        'EmployeeNumber': '员工编号',
        'EnvironmentSatisfaction': '环境满意度',
        'Gender': '性别',
        'JobInvolvement': '工作投入度',
        'JobLevel': '职位级别',
        'JobRole': '职位角色',
        'JobSatisfaction': '工作满意度',
        'MaritalStatus': '婚姻状况',
        'MonthlyIncome': '月收入',
        'NumCompaniesWorked': '工作公司数',
        'Over18': '是否成年',
        'OverTime': '是否加班',
        'PercentSalaryHike': '涨薪幅度',
        'PerformanceRating': '绩效评级',
        'RelationshipSatisfaction': '关系满意度',
        'StandardHours': '标准工时',
        'StockOptionLevel': '股票期权等级',
        'TotalWorkingYears': '总工作年数',
        'TrainingTimesLastYear': '去年培训次数',
        'WorkLifeBalance': '工作生活平衡',
        'YearsAtCompany': '司龄',
        'YearsInCurrentRole': '当前职位年数',
        'YearsSinceLastPromotion': '距上次晋升年数',
        'YearsWithCurrManager': '与现任经理共事年数'
    }

    # 方法1：读取数据时重命名,获取数据源
    data = pd.read_csv(path).rename(columns=column_mapping)

    # # 1.获取数据源
    # data = pd.read_csv(path)
    # # 2.时间格式化
    # data['time'] = pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    # # 3.按时间升序排序
    # data.sort_values(by='time', inplace=True)
    # # 4.去重
    # data.drop_duplicates(inplace=True)
    return data


def mean_absolute_percentage_error(y_true, y_pred):
    """
    低版本的sklearn没有MAPE的计算方法，需要自己定义，高版本的可以直接调用
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: MAPE（平均绝对百分比误差）
    """
    n = len(y_true)
    if len(y_pred) != n:
        raise ValueError("y_true and y_pred have different number of output ")
    abs_percentage_error = np.abs((y_true - y_pred) / y_true)
    return np.sum(abs_percentage_error) / n * 100
