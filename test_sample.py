from ouster import client, pcap
from contextlib import closing
from more_itertools import nth
import matplotlib.pyplot as plt

pcap_path = '/Users/theodorjonsson/GithubProjects/OS0-128_Rev-06_fw23_Urban-Drive_Dual-Returns/OS0-128_Rev-06_fw23_Urban-Drive_Dual-Returns.pcap'
metadata_path = '/Users/theodorjonsson/GithubProjects/OS0-128_Rev-06_fw23_Urban-Drive_Dual-Returns/OS0-128_Rev-06_fw23_Urban-Drive_Dual-Returns.json'
with open(metadata_path,'r')as file:
    info = client.SensorInfo(file.read())
source = pcap.Pcap(pcap_path,info)
with closing(client.Scans(source)) as scans:
    scan = nth(scans, 50)

range_field = scan.field(client.ChanField.RANGE)
range_img = client.destagger(info, range_field)
#plt.imshow(range_img[:, 640:1024], resample=False)
print(type(range_img))
#plt.axis('off')
#plt.show()