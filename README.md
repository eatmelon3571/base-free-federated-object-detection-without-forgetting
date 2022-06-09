### base-free-federated-object-detection-without-forgetting

### 数据集下载方式
下载方式一
https://www.nuscenes.org/nuimages
Downloads
All/Samples(15.27GB)

方式二
[Asia]
https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/nuimages-v1.0/nuimages-v1.0-all-samples.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=VLOdDskmSO9nuXUVgdlACVQeDZU%3D&Expires=1655103653

[US]
https://s3.amazonaws.com/data.nuscenes.org/public/nuimages-v1.0/nuimages-v1.0-all-samples.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=QIEWgJS6dXrhL5o%2Fu5LeABqAdQ4%3D&Expires=1655103680

### 数据集生成方式
1  下载完成后解压，放入与项目根目录/data下

2  convert_vehicle_label.py

3  split_vehicle_data_34_1800.py

4  split_base_to_all_and_novel.py



