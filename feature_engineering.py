import numpy as np
from pyspark import SQLContext, SparkContext
import pyspark.sql.functions as functions
from pyspark.sql.types import *
import findspark


findspark.init()
# pyspark
sparkc = SparkContext('local')
sqlc = SQLContext(sparkc)

train = sqlc.read.csv('data/train.csv', sep='\t', header=True)
test = sqlc.read.csv('data/test.csv', sep='\t', header=True)


# ======================================ADJUSTING====================================== #
# 1. removing
# We should omit columns that do not appear in test data set, simply removing.

# """  train.columns = 
#      Index(['Row', 'Anon Student Id', 'Problem Hierarchy', 'Problem Name',
#        'Problem View', 'Step Name', 'Step Start Time',
#        'First Transaction Time', 'Correct Transaction Time', 'Step End Time',
#        'Step Duration (sec)', 'Correct Step Duration (sec)',
#        'Error Step Duration (sec)', 'Correct First Attempt', 'Incorrects',
#        'Hints', 'Corrects', 'KC(Default)', 'Opportunity(Default)'],
#       dtype='object') """

removing_cols = ['Row', 'Step Start time', 'First Transaction Time', 
                    'Correct Transaction Time', 'Step End Time', 'Step Duration (sec)',
                   'Correct Step Duration (sec)', 'Error Step Duration (sec)', 
                   'Incorrects', 'Hints', 'Corrects']

train = train.drop(*removing_cols)
test = test.drop(*removing_cols)

# 2. dividing before encoding
# Divide column 'Problem Hierarchy' into two columns: 'Problem Unit' and 'Problem Section'.

# """ Problem Hierarchy               => Problem Unit & Problem Section
#     Unit CTA1_06, Section CTA1_06-3 => Unit CTA1_06 & Section CTA1_06-3 """

def get_first_attr(str):
    return str.split(',')[0]

def get_second_attr(str):
    return str.split(',')[1]

# use user define function
udfget_first_attr = functions.udf(get_first_attr, StringType())
udfget_second_attr = functions.udf(get_second_attr, StringType())

train = train.withColumn('Problem Unit', udfget_first_attr('Problem Hierarchy'))
train = train.withColumn('Problem Section', udfget_second_attr('Problem Hierarchy'))
test = test.withColumn('Problem Unit', udfget_first_attr('Problem Hierarchy'))
test = test.withColumn('Problem Section', udfget_second_attr('Problem Hierarchy'))

removing_cols = ['Problem Hierarchy']
train = train.drop(*removing_cols)
test = test.drop(*removing_cols)

# adjusting finished

# ======================================ENCODING====================================== #
# naive encoding of specific string columns
def encode(column):
    global train, test
    string_rows = train.union(test).select(column).distinct().collect()
    string2int = {}
    for i, row in enumerate(string_rows):
        string2int[row[column]] = i

    def col2int(str):
        return string2int[str]

    udfcol2int = functions.udf(col2int, IntegerType())

    train = train.withColumn(column, udfcol2int(column))    # replace
    test = test.withColumn(column, udfcol2int(column))


encoding_cols = ['Anon Student Id', 'Problem Name',
                    'Problem Unit', 'Problem Section', 
                    'Step Name']
for _ in encoding_cols:
    encode(_)

# encoding finished

# ======================================dealing with ~~====================================== #
# 1. take the average of opportunities, drop opportunity

def avgOp(str):
    _sum = 0.0
    _avg = 0.0
    if not str:
        return 0.0
    else:
        ops = str.split('~~')
        counter = 0
        for i, _ in enumerate(ops):
            _sum += eval(_)
            counter = counter + 1
        _avg = float(_sum / counter)
        return _avg


udfavgOp = functions.udf(avgOp, FloatType())

train = train.withColumn('Opportunity Average', udfavgOp('Opportunity(Default)'))
train = train.drop('Opportunity(Default)')
test = test.withColumn('Opportunity Average', udfavgOp('Opportunity(Default)'))
test = test.drop('Opportunity(Default)')


# 2. drop KC
# 2.1 add the number of KC first

def countKC(str):
    if not str:
        return 0
    else:
        return 1 + str.count('~~')


udfcountKC = functions.udf(countKC, IntegerType())

train = train.withColumn('KC Count', udfcountKC('KC(Default)'))
test = test.withColumn('KC Count', udfcountKC('KC(Default)'))

# 2.2 add new features

def add_new_features():
    global train, test


    benchmark = train.filter(train['Correct First Attempt'] == '1')

    # Personal Correct First Attempt Count(Personal CFAC)
    # 1)
    def personal_cfac():
        global train, test
        _pc_string2int = {}
        _all_pc = train.groupBy('Anon Student Id').count().collect()
        for _ in _all_pc:
            _pc_string2int[_['Anon Student Id']] = 0
        _some_pc = benchmark.groupBy('Anon Student Id').count().collect()
        for _ in _some_pc:   # replace
            _pc_string2int[_['Anon Student Id']] = _['count']

        _sum = 0
        for _ in _pc_string2int.keys():
            _sum += _pc_string2int[_]

        _avg = float(_sum / len(_all_pc))

        def id2count(_id):
            if _id in _pc_string2int.keys():
                return float(_pc_string2int[_id])
            else:
                return _avg

        udfid2count = functions.udf(id2count, FloatType())
        train = train.withColumn('Personal CFAC', udfid2count('Anon Student Id'))
        test = test.withColumn('Personal CFAC', udfid2count('Anon Student Id'))


    personal_cfac()
    # Personal CFAC finished

    # Personal Correct First Attempt Rate(Personal CFAR)
    # 2)
    def personal_cfar():
        global train, test
        _rate_temp = {}
        _rate_string2int = {}
        _some_pc = benchmark.groupBy('Anon Student Id').count().collect()
        _all_pc = train.groupBy('Anon Student Id').count().collect()
        
        for _ in _all_pc:
            _rate_temp[_['Anon Student Id']] = _['count']
        # print(_rate_temp)
        for _ in _all_pc:
            _rate_string2int[_['Anon Student Id']] = 0
        for _ in _some_pc:  # replace
            # print(_['count'] / _rate_temp[_['Anon Student Id']])
            _rate_string2int[_['Anon Student Id']] = _['count'] / _rate_temp[_['Anon Student Id']]
        _sum = 0
        for key in _rate_string2int.keys():
            _sum += _rate_string2int[key]
        _avg = float(_sum/len(_all_pc))

        def id2rate(id):
            if id in _rate_string2int.keys():
                return float(_rate_string2int[id])
            else:
                return _avg

        udfid2rate = functions.udf(id2rate, FloatType())
        train = train.withColumn(
            'Personal CFAR', udfid2rate('Anon Student Id'))
        test = test.withColumn('Personal CFAR', udfid2rate('Anon Student Id'))


    personal_cfar()
    # Personal CFAR finished.

    # Problem Correct First Attempt Rate(Problem CFAR)
    # 3)
    def problem_cfar():
        global train, test
        _rate_temp = {}
        _rate_string2int = {}
        _some_pc = benchmark.groupBy('Problem Name').count().collect()
        _all_pc = train.groupBy('Problem Name').count().collect()
        
        for _ in _all_pc:
            _rate_temp[_['Problem Name']] = _['count']
        # print(_rate_temp)
        for _ in _all_pc:
            _rate_string2int[_['Problem Name']] = 0
        for _ in _some_pc:  # replace
            # print(_['count'] / _rate_temp[_['Anon Student Id']])
            _rate_string2int[_['Problem Name']] = _['count'] / _rate_temp[_['Problem Name']]
        _sum = 0
        for key in _rate_string2int.keys():
            _sum += _rate_string2int[key]
        _avg = float(_sum/len(_all_pc))

        
        def id2rate(id):
            if id in _rate_string2int.keys():
                return float(_rate_string2int[id])
            else:
                return _avg

        udfid2rate = functions.udf(id2rate, FloatType())
        train = train.withColumn('Problem CFAR', udfid2rate('Problem Name'))
        test = test.withColumn('Problem CFAR', udfid2rate('Problem Name'))


    problem_cfar()
    # Problem CFAR finished.

    # Unit Correct First Attempt Rate(Unit CFAR)
    # 4)
    def unit_cfar():
        global train, test
        _rate_temp = {}
        _rate_string2int = {}
        _some_pc = benchmark.groupBy('Problem Unit').count().collect()
        _all_pc = train.groupBy('Problem Unit').count().collect()
        
        for _ in _all_pc:
            _rate_temp[_['Problem Unit']] = _['count']
        # print(_rate_temp)
        for _ in _all_pc:
            _rate_string2int[_['Problem Unit']] = 0
        for _ in _some_pc:  # replace
            # print(_['count'] / _rate_temp[_['Anon Student Id']])
            _rate_string2int[_['Problem Unit']] = _['count'] / _rate_temp[_['Problem Unit']]
        _sum = 0
        for key in _rate_string2int.keys():
            _sum += _rate_string2int[key]
        _avg = float(_sum/len(_all_pc))

        
        def id2rate(id):
            if id in _rate_string2int.keys():
                return float(_rate_string2int[id])
            else:
                return _avg

        udfid2rate = functions.udf(id2rate, FloatType())
        train = train.withColumn('Unit CFAR', udfid2rate('Problem Unit'))
        test = test.withColumn('Unit CFAR', udfid2rate('Problem Unit'))


    unit_cfar()
    # Unit CFAR finished.

    # Section Correct First Attempt Rate(Section CFAR)
    # 5)
    def section_cfar():
        global train, test
        _rate_temp = {}
        _rate_string2int = {}
        _some_pc = benchmark.groupBy('Problem Section').count().collect()
        _all_pc = train.groupBy('Problem Section').count().collect()
        
        for _ in _all_pc:
            _rate_temp[_['Problem Section']] = _['count']
        # print(_rate_temp)
        for _ in _all_pc:
            _rate_string2int[_['Problem Section']] = 0
        for _ in _some_pc:  # replace
            # print(_['count'] / _rate_temp[_['Anon Student Id']])
            _rate_string2int[_['Problem Section']] = _['count'] / _rate_temp[_['Problem Section']]
        _sum = 0
        for key in _rate_string2int.keys():
            _sum += _rate_string2int[key]
        _avg = float(_sum/len(_all_pc))

        
        def id2rate(id):
            if id in _rate_string2int.keys():
                return float(_rate_string2int[id])
            else:
                return _avg

        udfid2rate = functions.udf(id2rate, FloatType())
        train = train.withColumn('Section CFAR', udfid2rate('Problem Section'))
        test = test.withColumn('Section CFAR', udfid2rate('Problem Section'))


    section_cfar()
    # Section CFAR finished

    # Step Correct First Attempt Rate(Step CFAR):
    # 6)
    def step_cfar():
        global train, test
        _rate_temp = {}
        _rate_string2int = {}
        _some_pc = benchmark.groupBy('Step Name').count().collect()
        _all_pc = train.groupBy('Step Name').count().collect()
        
        for _ in _all_pc:
            _rate_temp[_['Step Name']] = _['count']
        # print(_rate_temp)
        for _ in _all_pc:
            _rate_string2int[_['Step Name']] = 0
        for _ in _some_pc:  # replace
            # print(_['count'] / _rate_temp[_['Anon Student Id']])
            _rate_string2int[_['Step Name']] = _['count'] / _rate_temp[_['Step Name']]
        _sum = 0
        for key in _rate_string2int.keys():
            _sum += _rate_string2int[key]
        _avg = float(_sum/len(_all_pc))

        
        def id2rate(id):
            if id in _rate_string2int.keys():
                return float(_rate_string2int[id])
            else:
                return _avg

        udfid2rate = functions.udf(id2rate, FloatType())
        train = train.withColumn('Step CFAR', udfid2rate('Step Name'))
        test = test.withColumn('Step CFAR', udfid2rate('Step Name'))


    step_cfar()
    # Step CFAR finished

    # KC Correct First Attempt Rate(KC CFAR)
    # 7)
    def kc_cfar():
        global train, test
        _rate_temp = {}
        _rate_string2int = {}
        _some_pc = benchmark.groupBy('KC(Default)').count().collect()
        _all_pc = train.groupBy('KC(Default)').count().collect()
        
        for _ in _all_pc:
            _rate_temp[_['KC(Default)']] = _['count']
        # print(_rate_temp)
        for _ in _all_pc:
            _rate_string2int[_['KC(Default)']] = 0
        for _ in _some_pc:  # replace
            # print(_['count'] / _rate_temp[_['Anon Student Id']])
            _rate_string2int[_['KC(Default)']] = _['count'] / _rate_temp[_['KC(Default)']]
        _sum = 0
        for key in _rate_string2int.keys():
            _sum += _rate_string2int[key]
        _avg = float(_sum/len(_all_pc))

        
        def id2rate(id):
            if id in _rate_string2int.keys():
                return float(_rate_string2int[id])
            else:
                return _avg

        udfid2rate = functions.udf(id2rate, FloatType())
        train = train.withColumn('KC CFAR', udfid2rate('KC(Default)'))
        test = test.withColumn('KC CFAR', udfid2rate('KC(Default)'))


    kc_cfar()
    train = train.drop('KC(Default)')
    test = test.drop('KC(Default)')
    # KC CFAR finished


add_new_features()

# ======================================export====================================== #

train.toPandas().to_csv('data/preprossed_train.csv',sep='\t', header=True, index=False)
test.toPandas().to_csv('data/preprossed_test.csv', sep='\t', header=True, index=False)

