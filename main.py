# from key_listener import RealTimeKeyListener
#
# real_time_listener = RealTimeKeyListener()

import nltk, os, json
sentence = 'testowe: /działanie.[] aplikacji,'
position = -10
__word_tokenizer = nltk.tokenize.RegexpTokenizer('[\s,:/\".;!?\n{}\[\]()-+=_~`@#$%^&*]', gaps=True)
print(__word_tokenizer.tokenize(sentence))
# print(sentence)
# print(sentence[:-1])

# destination_json_file_path = "C:/Users/user/Desktop/destination_file.json"
#
# def read_json_file(path_to_file):
#     with open(path_to_file, 'r') as f:
#         data = json.load(f)
#     return data
#
# def add_dict_list_to_json_file(path_to_file, key, obj):
#     # check if is empty
#     if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
#         data = read_json_file(path_to_file)
#         open(path_to_file, 'w').close()
#         file = open(path_to_file, 'a+')
#         if isinstance(obj, dict) and obj.keys() and key in data.keys():
#             for k in obj.keys():
#                 if k not in data[key].keys():
#                     data[key][k] = obj[k]
#                 elif k in data[key].keys() and (isinstance(data[key][k], int) or isinstance(data[key][k], float)):
#                     data[key][k] += obj[k]
#
#         elif key in data.keys() and isinstance(obj, list) and obj and isinstance(data[key], list):
#             data[key].extend(obj)
#         else:
#             data[key] = obj
#             file.seek(0)
#         json.dump(data, file, indent=4)
#         file.close()

