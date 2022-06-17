from mimetypes import init
from ouster import client, pcap
from contextlib import closing
import cv2
#import matplotlib.pyplot as plt
from more_itertools import time_limited,nth
import numpy as np
from timeit import default_timer as timer
from tqdm import tqdm
import time
from datetime import datetime as dt
import open3d as o3d
import os

def sensor_config(hostname = 'os-122107000535.local',lidar_port = 7502, imu_port = 7503): 
    """
    Set sensor configuration.
    @param hostname: sensor hostname
    @param lidar_port: lidar port
    @param imu_port: imu port

    @return: Sensor Config Object.
    """
   
    print("Configuring sensor.")
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

def record_lidar_seq(
                    config=None,
                    fileformat="npy",
                    lidar_folder_path="../lidar_scans",
                    scan_name = str(f"Ouster_{dt.now().strftime('%Y%m%d_%H%M%S')}"),
                    set_frames = False,
                    seq_length = 1.0,
                    hostname = 'os-122107000535.local',
                    lidar_port = 7502,
                    imu_port = 7503
                    ):
    """
    Record lidar data for n_seconds and save to local generated path..
    @param config: SensorConfig object
    @param fileformat: string (nly, npy, pcap)
    @param lidar_folder_path: string (path to lidar folder)
    @param scan_name: string (name of scan)
    @param set_frames: bool if True will only record a fixed number of frames. If False will record seq_lenght seconds
    @param seq_length: float (number of frames to record) if set_frames is False else number of frames to record.
    @param hostname: string
    @param lidar_port: int (default 7502)
    @param imu_port: int (default 7503)


    """
    if config is None:
        try:
            [config,_] = sensor_config(hostname=hostname,lidar_port=lidar_port,imu_port=imu_port)
        except:
            raise Exception("Could not connect to sensor.")
    if not os.path.exists(lidar_folder_path):
        os.makedirs(lidar_folder_path)
    scan_path = f"{lidar_folder_path}/{scan_name}/"
    if not os.path.exists(scan_path):
        os.mkdir(f"{lidar_folder_path}/{scan_name}")
    # connect to sensor and record lidar/imu packets
    frames = 0
    full_seq = []
    if fileformat == "npy":
        
        with closing(client.Scans.stream(hostname, lidar_port,
                                    complete=False)) as stream:
            print(f"Record lidar data to npy files {seq_length}:")
            if not set_frames:
                start = time.time()
            else:
                seq_length = int(seq_length)

            start = time.time()
            with tqdm(total=seq_length) as pbar:
                for seq,scan in enumerate(stream,0):
                    xyz = get_xyz(stream,scan)
                    #print(f"xyz shape: {xyz.shape}")
                    signal = get_signal_reflection(stream,scan)
                    #print(f"xyz shape: {xyz.shape}")
                    xyzr = convert_to_xyzr(xyz,signal)
                    #print(f"xyz shape: {xyz.shape}")
                    comp_xyzr = compress_mid_dim(xyzr)
                    #print(f"comp_xyzr shape: {comp_xyzr.shape}")
                    full_seq.append(comp_xyzr)
                    save_to_nly(comp_xyzr,scan_path,seq)
                    #print(f"Time: {time.time()-start} frame {seq}")
                    if not set_frames:
                        pbar.update(((time.time()-start)/seq_length))
                        if time.time()-start>seq_length:
                            frames = seq + 1
                            break
                    if seq>seq_length and set_frames:
                        pbar.update()
                        break
            #full_seq = np.asarray(full_seq)
            print(f"Full Seq shape: {full_seq.shape}")
            #save_to_nly(full_seq,f"{scan_path+"_FS"}",0)
    
    elif fileformat == "pcap":
        with closing(client.Sensor(hostname, lidar_port, imu_port,
                                   buf_size=640)) as source:

            # make a descriptive filename for metadata/pcap files

            time_part = dt.now().strftime("%Y%m%d_%H%M%S")
            meta = source.metadata
            fname_base = f"{meta.prod_line}_{meta.sn}_{meta.mode}_{time_part}"

            print(f"Saving sensor metadata to: {fname_base}.json")
            source.write_metadata(f"{fname_base}.json")

            print(f"Writing to: {fname_base}.pcap (Ctrl-C to stop early)")
            source_it = time_limited(seq_length, source)
            n_packets = pcap.record(source_it, f"{fname_base}.pcap")
            print(f"Captured {n_packets} packets")
    return scan_path,frames

def save_to_nly(xyzr,seq_path,seq):
    """
    Save lidar data to nly file.
    @param xyzr: numpy array
    @param scan_folder: string (name of scan folder)
    @param seq: int (sequence number)
    """
    filename = f"{seq_path}{seq}"
    #print(f"Saving to: {filename}")
    np.save(filename,xyzr)


def play_back_recording(seq_dir_path):
    """
    Play back lidar data from nly files.
    @param seq_dir_path: string (path to lidar scan folder) containing .npy
    """
    if not os.path.exists(seq_dir_path):
        raise FileNotFoundError(f"{seq_dir_path} does not exist")
    print(f"Playing back lidar data from: {seq_dir_path}")
    for i,files in enumerate(os.listdir(seq_dir_path)):
        if files.endswith(".npy"):
            xyzr = np.load(f"{seq_dir_path}/{files}")
            if len(xyzr.shape)==3:
                xyzr = compress_mid_dim(xyzr)
            xyz = xyzr[:,:3]
            signal = xyzr[:,3]
            #print(f"xyzr shape: {xyzr.shape}")
            if i == 0:
                vis, geo = initialize_o3d_plot(xyzr)
            else:
                update_open3d_live(geo, xyzr,vis)
            #time.sleep(0.1)
    
def play_back_recording_multifile(seq_dir_path):
    """
    Play back lidar data from nly files.
    @param seq_dir_path: string (path to lidar scan folder) containing .npy
    """
    if not os.path.exists(seq_dir_path):
        raise FileNotFoundError(f"{seq_dir_path} does not exist")
    print(f"Playing back lidar data from: {seq_dir_path}")
    for i,files in enumerate(os.listdir(seq_dir_path)):
        if files.endswith(".npy"):
            print(f"Plotting lidar data from file: {files}")
            full_seq = np.load(f"{seq_dir_path}/{files}")
            for j,xyzr in enumerate(full_seq):
                xyzr = np.squeeze(xyzr)
                xyzr = compress_mid_dim(xyzr) if len(xyzr.shape)==3 else xyzr
                #print(f"xyzr shape: {xyzr.shape}")
                if j == 0:
                    vis, geo = initialize_o3d_plot(xyzr)
                else:
                    update_open3d_live(geo, xyzr,vis)
                    time.sleep(0.05)
            #time.sleep(0.1)
    
def process_lidar_scan(scan_path=None):
    """
    Process lidar scan data.
    @param scan_path: string (path to lidar scan folder) containing .npy
    """
    if not os.path.exists(scan_path):
        raise FileNotFoundError(f"{scan_path} does not exist")
    print(f"Processing lidar data from: {scan_path}")
    for i,files in enumerate(os.listdir(scan_path)):
        if files.endswith("FS.npy"):
            print(f"Processing lidar data from file: {files}")
            scn = np.load(f"{scan_path}/{files}")
def stream_live(config=None,hostname = 'os-122107000535.local',lidar_port = 7502, imu_port = 7503):
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
        # First frame to initialize visualizer
        first_scan = next(iter(stream))
        xyz = get_xyz(stream,first_scan)
        signal = get_signal_reflection(stream,first_scan)
        xyzr = convert_to_xyzr(xyz,signal)
        comp_xyzr = compress_mid_dim(xyzr)
        vis, geo = initialize_o3d_plot(comp_xyzr)
        # start the stream
        for i,scan in enumerate(stream):
            # uncomment if you'd like to see frame id printed
            # print("frame id: {} ".format(scan.frame_id))
            signal = get_signal_reflection(stream,scan)
            xyz = get_xyz(stream,scan)
            xyzr = convert_to_xyzr(xyz,signal)
            comp_xyzr = compress_mid_dim(xyzr)
            update_open3d_live(geo,comp_xyzr,vis)
            # print(f"T: {end-start} s")
            if i>1000:
                break
def convert_to_xyzr(xyz,signal):
    """
    Convert lidar data to xyzr.
    @param xyz: numpy array of xyz data
    @param signal: numpy array of signal data

    @return: xyzr numpy array
    """
    if xyz is None:
        raise ValueError("xyz is None before concatenation with signal.")
    if signal is None:
        raise ValueError("signal is None before concatenation with xyz.")
    if len(xyz.shape) < 3 or xyz.shape[-1] != 3:
        raise ValueError("xyz is not (channels,beams,3)")
    if len(signal.shape) < 3 or signal.shape[-1] != 1:
        #"Signals should same as xyz except final dim: (n_scans, n_points, 1)")
        signal = np.expand_dims(signal,axis=-1)
    #print("xyz.shape: {}".format(xyz.shape))
    #print("signal.shape: {}".format(signal.shape))
    try:
        xyzr = np.concatenate((xyz,signal),axis=-1)
    except:
        print("Different shapes for xyz and signal. xyz: {}, signal: {}".format(xyz.shape,signal.shape))
        return None
    return xyzr

def compress_mid_dim(data):
    """
    Compress mid dim of xyz,xyzr or signal to create (bs,n_points, 1 3 or 4).
    """
    if len(data.shape) < 3:
        raise ValueError(f"xyz should be: (channels, beams, 3 or 4). got {data.shape}.")
    return data.reshape(-1,data.shape[-1])

def get_signal_reflection(source,scan):
    """
    Get signal reflection from scan.
    @param scan: Scan object
    @param source: Source of data

    @return: numpy array of signal reflection
    """
    if scan is None:
        raise ValueError("scan is None")
    if source is None:
        raise ValueError("source is None")
    try:
        return client.destagger(source.metadata,
                            scan.field(client.ChanField.REFLECTIVITY))
    except:
        print("Error getting signal reflection.")
        return None
def get_xyz(source,scan):
    """
    Get xyz data from scan.
    @param source: Source of data
    @param scan: Scan object
    @return: numpy array of xyz data
    """
    if source is None:
        raise ValueError("source is None")
    xyzlut = client.XYZLut(source.metadata)
    return xyzlut(scan)

#def plot_lidar_example(xyz):
#    """
#    Plot lidar example from xyz data.
#    @param xyz: numpy array of xyz data
#    """
#    [x, y, z] = [c.flatten() for c in np.dsplit(xyz, 3)]
#    ax = plt.axes(projection='3d')
#    r = 5
#    ax.set_xlim3d([-r, r])
#    ax.set_ylim3d([-r, r])
#    ax.set_zlim3d([-r/2, r/2])
#    plt.axis('off')
#    z_col = np.minimum(np.absolute(z), 5)
#    ax.scatter(x, y, z, c=z_col, s=0.2)
#    plt.show()

def get_single_example():
    """
    Get a single example from the lidar data.

    """
    metadata_path = "/Users/theodorjonsson/GithubProjects/LidarStuff/RecordedData/OS0/json/OS0-128_Rev-06_fw23_Urban-Drive_Dual-Returns.json"
    pcap_path = "/Users/theodorjonsson/GithubProjects/LidarStuff/RecordedData/OS0/pcap/OS0-128_Rev-06_fw23_Urban-Drive_Dual-Returns.pcap"
    #pcap_path = '/Users/theodorjonsson/GithubProjects/ExampleData/OS0-128_Rev-06_fw23_Urban-Drive_Dual-Returns.pcap'
    #metadata_path = '/Users/theodorjonsson/GithubProjects/ExampleData/OS0-128_Rev-06_fw23_Urban-Drive_Dual-Returns.json'
    with open(metadata_path,'r') as file:
        info = client.SensorInfo(file.read())
    source = pcap.Pcap(pcap_path,info)
    with closing(client.Scans(source)) as scans:
        scan = nth(scans, 50) # Five second scan (50Rot/10Hz)
    return scan,source
def initialize_o3d_plot(xyzr):
    """
    Initialize o3d plot.
    @param xyzr: numpy array of xyzr data to initailize plot with.

    @return: o3d.visualization.Visualizer object
    """
    xyzr = xyzr[0] if len(xyzr.shape) > 2 else xyzr
    xyz = xyzr[:,:3]
    signal = xyzr[:,3:]
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geo = o3d.geometry.PointCloud()
    #print("xyz.shape: {}".format(xyz.shape))
    geo.points = o3d.utility.Vector3dVector(xyz)
    vis.add_geometry(geo)
    vis.run()
    return vis,geo

def update_open3d_live(geo,xyzr,vis):
    """
    Plot lidar data live using open3d.
    @param xyzr: numpy array of xyzr data
    """
    xyzr = xyzr[0] if len(xyzr.shape) > 2 else xyzr
    xyz = xyzr[:,:3]
    signal = xyzr[:,3:]
    #print("xyz.shape: {}".format(xyz.shape))
    geo.points = o3d.utility.Vector3dVector(xyz)
    vis.update_geometry(geo)
    vis.poll_events()
    vis.update_renderer()
    
    
def plot_open3d_pc(xyzr):
    """
    Plot lidar example from xyzr data.
    @param xyzr: numpy array of xyzr data. Each cloud is stacked in dim=0. Iterate over dim=0 to visualize all clouds.
    """
    xyz = xyzr[:,:,:3] 
    signal = xyzr[:,:,3:] # 3: for broadcasting
    print("Shape of xyz: {}".format(xyz.shape))
    print("Shape of signal: {}".format(signal.shape))
   
    vis,geo = initialize_o3d_plot(xyz[0])
    for i,cloud in enumerate(xyz[1:],1):
        geo.points = o3d.utility.Vector3dVector(cloud)
        geo.colors = o3d.utility.Vector3dVector(np.zeros((xyzr.shape[1],3))+signal[i,:,:]) # Not finished
        vis.update_geometry(geo)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1) # Time between clouds

if __name__ == "__main__":
    #filename,_ = record_lidar_seq(seq_length=5)
    filename = "lidar_scans/Ouster_20220617_100040"
    print("filename: {}".format(filename))
    play_back_recording_multifile(filename)
    #scan,source = get_single_example()
    #xyz = get_xyz(source,scan)
    #signal = get_signal_reflection(scan,source)
    #xyzr = convert_to_xyzr(xyz,signal)
    #plot_open3d_pc(compress_mid_dim(xyzr))
