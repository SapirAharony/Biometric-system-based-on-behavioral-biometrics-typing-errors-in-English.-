import system_features_extractor, os

################################## exctract data from original files


source_file_dir = 'C:\\Users\\user\\Desktop\\inz_wyniki'
dest_file_dir = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files'

for file in os.listdir(source_file_dir):
    if file[-4:] == 'json':
        print(file, end=' ')
        system_features_extractor.extract_data(source_file_dir + '\\' + file, dest_file_dir + '\\done_' + file, file[:-5].capitalize())


import string_metrics

print(string_metrics.Distances('abba', 'abab').operations[0].__dict__['ml_repr'])

print(string_metrics.Distances('abba', 'abb').operations[0].__dict__['ml_repr'])






