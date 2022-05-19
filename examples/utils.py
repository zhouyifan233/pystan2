import re
import numpy as np


def verifyDataType(model, data):
    model_str = model.model_code
    model_str = re.sub('[#|//].*', '', model_str)
    data_patch = re.search('data[ ]*{([^{}]*)}', model_str)
    data_str = data_patch.group(1)
    data_lines = data_str.split('\n')
    var_type_dic = {}
    single_val_to_array = {}     # for a special case that we need to convert a single value to an array
    for line in data_lines:
        valid_line = re.search('(.*);', line)
        if valid_line:
            single_val_to_array_ = False
            valid_line = valid_line.group(1)
            # pull out size. value in []
            size_part = re.search('\[([^\[\]]*)\]', valid_line)
            if size_part is not None:
                valid_line_1 = re.sub('\[([^\[\]]*)\]', '', valid_line)
                if size_part.group(1) == '1':
                    single_val_to_array_ = True
            else:
                valid_line_1 = valid_line
            # pull out range. value in <>
            range_str = re.search('\<([^\<\>]*)\>', valid_line_1)
            if range_str is not None:
                valid_line_2 = re.sub('\<([^\<\>]*)\>', '', valid_line)
            else:
                valid_line_2 = valid_line_1
            sep_line = re.search('[ ]*([^ ]*)[ ]*([^ \[\]]*)', valid_line_2)
            if sep_line:
                type_str = sep_line.group(1)
                var_str = sep_line.group(2)
                var_type_dic[var_str] = type_str
                single_val_to_array[var_str] = single_val_to_array_
    for data_key in data:
        if data_key in var_type_dic:
            claimed_type = var_type_dic[data_key]
            if claimed_type.startswith('int'):
                data[data_key] = np.int32(data[data_key])
            if single_val_to_array[data_key]:
                data[data_key] = np.array([data[data_key]])
        else:
            print("verify data type failed!" + data_key + " is not in the stan-model file...")

    return data


def getParameterNames(model):
    model_str = model.model_code
    model_str = re.sub('[#|//].*', '', model_str)
    model_str = model_str.strip()
    data_patch = re.search('parameters[ ]*{([^{}]*)}', model_str)
    if data_patch is not None:
        data_str = data_patch.group(1)
        data_lines = data_str.split('\n')
        var_type_dic = {}
        parameter_names = []
        for line in data_lines:
            line = re.sub('(//.*)', '', line)
            valid_line = re.search('(.*);', line)
            if valid_line:
                valid_line = valid_line.group(1)
                # pull out size in []
                size_part = re.search('\[([^\[\]]*)\]', valid_line)
                if size_part is not None:
                    valid_line = re.sub('\[([^\[\]]*)\]', '', valid_line)
                range_str = re.search('\<([^\<\>]*)\>', valid_line)
                if range_str is not None:
                    valid_line = re.sub('\<([^\<\>]*)\>', '', valid_line)
                sep_line = re.search('[ ]*([^ ]*)[ ]*([^ \[\]]*)', valid_line)
                if sep_line:
                    type_str = sep_line.group(1)
                    var_str = sep_line.group(2)
                    var_type_dic[var_str] = type_str
                    parameter_names.append(var_str)
    else:
        parameter_names = None

    return parameter_names

