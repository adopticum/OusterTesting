from ouster import client, pcap
from contextlib import closing
import cv2
import matplotlib.pyplot as plt
from more_itertools import time_limited,nth
import numpy as np
from timeit import default_timer as timer
from datetime import datetime
def sensor_config(hostname = 'os-122107000535.local',lidar_port = 7502, imu_port = 7503): 
    """
    Set sensor configuration.
    @param hostname: sensor hostname
    @param lidar_port: lidar port
    @param imu_port: imu port

    @return: Sensor Config Object.
    """
    # establish sensor connection
    config = client.SensorConfig()
    # set the values that you need: see sensor documentation for param meanings
    config.operating_mode = client.OperatingMode.OPERATING_NORMAL
    config.lidar_mode = client.LidarMode.MODE_1024x10
    config.udp_port_lidar = lidar_port
    config.udp_port_imu = imu_port

    # set the config on sensor, using appropriate flags
    client.set_config(hostname, config, persist=True, udp_dest_auto=True)
    return [config, hostname]

def record_lidar(config,n_seconds = 2,hostname = 'os-122107000535.local',lidar_port = 7502, imu_port = 7503):
    if config is None:
        [config,_] = sensor_config(hostname=hostname,lidar_port=lidar_port,imu_port=imu_port)
    # connect to sensor and record lidar/imu packets
    with closing(client.Sensor(hostname, lidar_port, imu_port,
                               buf_size=640)) as source:
    
        # make a descriptive filename for metadata/pcap files
        time_part = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta = source.metadata
        fname_base = f"{meta.prod_line}_{meta.sn}_{meta.mode}_{time_part}"
    
        print(f"Saving sensor metadata to: {fname_base}.json")
        source.write_metadata(f"{fname_base}.json")
    
        print(f"Writing to: {fname_base}.pcap (Ctrl-C to stop early)")
        source_it = time_limited(n_seconds, source)
        n_packets = pcap.record(source_it, f"{fname_base}.pcap")
        print(f"Captured {n_packets} packets")
def stream_live(config,hostname = 'os-122107000535.local',lidar_port = 7502, imu_port = 7503):
    """
    Stream Live from sensor belonging to hostname, given a specified config.
    @param config: SensorConfig object
    @param hostname: string
    @param lidar_port: int (default 7502)
    @param imu_port: int (default 7503)
    """
    if config is None:
        config = sensor_config(hostname=hostname,lidar_port=lidar_port,imu_port=imu_port)
    # create a stream object
    print("Start Lidar Stream:")
    with closing(client.Scans.stream(hostname, lidar_port,
                                    complete=False)) as stream:
        for i,scan in enumerate(stream):
            # uncomment if you'd like to see frame id printed
            # print("frame id: {} ".format(scan.frame_id))
            start = timer()
            signal = client.destagger(stream.metadata,
                                        scan.field(client.ChanField.REFLECTIVITY))
            xyzlut = client.XYZLut(stream.metadata)
            xyz = xyzlut(scan)
            print(np.shape(signal))
            print(np.shape(xyz))
            end = timer()
            # print(f"T: {end-start} s")
            #signal = (signal / np.max(signal) * 255).astype(np.uint8)
            #cv2.imshow("scaled signal", signal)
            #key = cv2.waitKey(1) & 0xFF
                # [doc-etag-live-plot-signal]
                # 27 is esc
            #if key == 27:
            #    show = False
            #    break
            if i>10:
                print(signal)
                print(xyz)
                break
if __name__ == "__main__":
    config,hostname = sensor_config()
    print(f"Sensor hostname: {hostname}")
    stream_live(config,hostname)
