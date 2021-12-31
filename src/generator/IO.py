import os

def image_txt(path):
    f = open("src/dataset/image.txt",'w') 
    path = path
    files =os.listdir(path) 
    files.sort(key=lambda d :int(d.split('_')[1]))
    for file_ in files:     
        if not os.path.isdir(path +file_):  
            f_name = str(file_.split(".")[0])
            f.write(f_name + '\n') 

if __name__ == "__main__":
    image_txt("src/dataset/images")
    