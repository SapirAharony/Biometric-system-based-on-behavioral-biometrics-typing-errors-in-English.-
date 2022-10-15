import system_features_extractor, os, string_metrics

################################## exctract data from original files


source_file_dir = 'C:\\Users\\user\\Desktop\\inz_wyniki'
dest_file_dir = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files'

# for file in os.listdir(source_file_dir):
#     if file[-4:] == 'json':
#         system_features_extractor.extract_data(source_file_dir + '\\' + file, dest_file_dir + '\\done_' + file, file[:-5].capitalize())

print(string_metrics.get_string_oprations('test', 'books'))