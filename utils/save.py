import os

def deleteAllFiles(directory = None):
    for file_name in os.listdir(directory):
        file_data = directory + "/" + file_name
        if os.path.isfile(file_data) == True:
            os.remove(file_data)
        else:
            deleteAllFiles(file_data)
            os.rmdir(file_data)

# Delete the directory including its all files
def delete_file(directory = None):
    deleteAllFiles(directory)
    os.rmdir(directory)

if __name__ == '__main__':
    delete_file("../temp")
