import os
import time

import cv2
import requests

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

proxy_url = os.environ.get('http_proxy', None)
if proxy_url:
    #! conda install pysocks
    proxies = {'http': proxy_url, }
else:
    proxies = None
url = 'https://gips2.baidu.com/it/u=828570294,3060139577&fm=3028&app=3028&f=JPEG&fmt=auto?w=1024&h=1024'
filename = 'food.jpg'
try:
    response = requests.get(url, proxies=proxies)
    response.raise_for_status()
    with open(filename, 'wb') as file:
        file.write(response.content)
    print(f"File downloaded successfully: {filename}")
except requests.RequestException as e:
    print(f"An error occurred: {e}")


image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
start_time = time.time()
edges_cpu = cv2.Canny(image, 100, 200)
cpu_time = time.time() - start_time

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(image)
    start_time = time.time()
    canny_gpu = cv2.cuda.createCannyEdgeDetector(100, 200)
    edges_gpu = canny_gpu.detect(gpu_image)
    edges_gpu = edges_gpu.download()
    gpu_time = time.time() - start_time
    print(f"CPU边缘检测时间: {cpu_time:.4f} 秒")
    print(f"GPU边缘检测时间: {gpu_time:.4f} 秒")
    cv2.imshow('原始图像', image)
    cv2.imshow('CPU边缘检测', edges_cpu)
    cv2.imshow('GPU边缘检测', edges_gpu)
else:
    print("OpenCV的CUDA版本未安装或CUDA未启用")
    cv2.imshow('原始图像', image)
    cv2.imshow('GPU边缘检测', edges_cpu)

os.remove(filename)
cv2.waitKey(0)
cv2.destroyAllWindows()
