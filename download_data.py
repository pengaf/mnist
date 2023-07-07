#pip install requests
import requests
import zipfile
import io

def download_and_extract(url, target_dir):
    # 下载文件
    response = requests.get(url)
    if response.status_code == 200:
        # 将文件保存到本地
        with open('temp.zip', 'wb') as file:
            file.write(response.content)
        
        # 解压文件
        with zipfile.ZipFile('temp.zip', 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        
        # 删除临时文件
        os.remove('temp.zip')
        print("文件下载和解压完成！")
    else:
        print("文件下载失败！")

download_and_extract('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'G:/mnist/')